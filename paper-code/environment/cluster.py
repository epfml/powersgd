#!/usr/bin/env python3

"""
This script manages a cluster on gcloud

# Prerequisites
1. Install the [gcloud cli](https://cloud.google.com/sdk/docs/quickstarts) and make sure you can access your account/project
2. Install `google-api-python-client` with pip on your local machine and make sure you are authorized. You can use `gcloud auth login`.
3. Make a service account 

# Create a snapshot on which all cluster nodes will be based
1. Create a n1-standard-32 instance named `basis` with 4 K80 GPUs, based on 'Ubuntu 18.04 LTS'
   It's good to have many CPUs because you'll need to build things.
   ```
   gcloud beta compute \
    --project=rank1-gradient-compression instances create basis \
    --zone=europe-west1-b \
    --machine-type=n1-standard-32 \
    --subnet=default \
    --network-tier=PREMIUM \
    --maintenance-policy=TERMINATE \
    --service-account=1048035866295-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=type=nvidia-tesla-k80,count=4 \
    --image=ubuntu-1804-bionic-v20200317 \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --boot-disk-device-name=basis \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --reservation-affinity=any
   ```
2. Initialize ssh access with `gcloud compute config-ssh`
3. `rsync -av files/ basis:`
4. Copy `jobmonitor.whl` to basis:install/
5. Run `~/setup.sh` on `basis` -- this will take more than an hour
6. Create a snapshot of the machine
   ```
   gcloud compute disks snapshot basis \
    --project=rank1-gradient-compression \
    --snapshot-names=cluster-template \
    --zone=europe-west1-b \
    --storage-location=eu
   ```
6b. Create an image from the snapshot
    ```
    gcloud compute images create cluster-template \
        --project=rank1-gradient-compression \
        --source-snapshot=cluster-template \
        --storage-location=europe-west1
    ```
7. Delete the basis machine:
   ```
   gcloud compute instances delete basis
   ``` 
8. Create the cluster with this file:
   ```
   ./cluster.py create -n 2
   ```

# 

"""

import argparse
import os
import subprocess
import tempfile
import threading
import time

import googleapiclient.discovery

DEFAULT_NAME = "cluster"
DEFAULT_PROJECT = "rank1-gradient-compression"
DEFAULT_NUM_WORKERS = 1
DEFAULT_NUM_GPUS_PER_WORKER = 1
DEFAULT_ZONE = "europe-west1-b"
DEFAULT_IMAGE = "cluster-template"
DEFAULT_GPU_TYPE = "nvidia-tesla-k80"
DEFAULT_MACHINE_TYPE = "n1-standard-4"
DEFAULT_CONTROLLER_MACHINE_TYPE = "n1-standard-1"
DEFAULT_SERVICE_ACCOUNT = "experiments@rank1-gradient-compression.iam.gserviceaccount.com"
DEFAULT_BUCKET_NAME = "decentralized-powersgd"
DEFAULT_PREEMPTIBLE = False


class Cluster:
    def __init__(
        self,
        name=DEFAULT_NAME,
        num_workers=DEFAULT_NUM_WORKERS,
        project=DEFAULT_PROJECT,
        zone=DEFAULT_ZONE,
        image=DEFAULT_IMAGE,
        machine_type=DEFAULT_MACHINE_TYPE,
        gpu_type=DEFAULT_GPU_TYPE,
        num_gpus_per_worker=DEFAULT_NUM_GPUS_PER_WORKER,
        controller_machine_type=DEFAULT_CONTROLLER_MACHINE_TYPE,
        service_account=DEFAULT_SERVICE_ACCOUNT,
        bucket_name=DEFAULT_BUCKET_NAME,
        is_preemptible=DEFAULT_PREEMPTIBLE,
        jobmonitor_whl="/Users/vogels/epfl/job-monitor/dist/jobmonitor-0.1-py3-none-any.whl",
    ):
        self.compute = googleapiclient.discovery.build("compute", "v1")
        self.name = name
        self.project = project
        self.zone = zone
        self.region = zone[:-2]
        self.image = image
        self.num_workers = num_workers
        self.controller_machine_type = controller_machine_type
        self.machine_type = machine_type
        self.service_account = service_account
        self.bucket_name = bucket_name
        self.gpu_type = gpu_type
        self.gpu_type_slurm = gpu_type.replace("nvidia-", "").replace("-", "_")
        self.num_gpus_per_worker = num_gpus_per_worker
        self.jobmonitor_whl = jobmonitor_whl
        self.is_preemptible = is_preemptible

        self.base_config = {
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    # "deviceName": self.controller_name(),
                    "type": "PERSISTENT",
                    "mode": "READ_WRITE",
                    "initializeParams": {
                        "sourceImage": f"projects/{self.project}/global/images/{self.image}",
                        # "sizeGb": "50",
                        "type": f"projects/{self.project}/zones/{self.zone}/diskTypes/pd-ssd",
                        "zone": f"projects/{self.project}/zones/{self.zone}",
                        "diskEncryptionKey": {},
                    },
                }
            ],
            "canIpForward": False,
            "networkInterfaces": [
                {
                    "subnetwork": f"projects/{self.project}/regions/{self.region}/subnetworks/default",
                    "accessConfigs": [
                        {
                            "kind": "compute#accessConfig",
                            "name": "External NAT",
                            "type": "ONE_TO_ONE_NAT",
                            "networkTier": "PREMIUM",
                        }
                    ],
                    "aliasIpRanges": [],
                }
            ],
            "scheduling": {
                "preemptible": self.is_preemptible,
                "onHostMaintenance": "TERMINATE",
                "automaticRestart": False,
                "nodeAffinities": [],
            },
            "deletionProtection": False,
            "reservationAffinity": {"consumeReservationType": "ANY_RESERVATION"},
            "description": "",
            "labels": {},
            "serviceAccounts": [
                {
                    "email": self.service_account,
                    "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
                }
            ],
        }

    def spin_up(self):
        """Ensures that all the nodes are up and slurm configs are setup correctly"""
        operations = []
        if not self.controller_exists():
            operations.append(self.create_controller_node())

        for worker in range(self.num_workers):
            if not self.worker_exists(worker):
                operations.append(self.create_worker_node(worker))
        self.wait_to_finish(operations)
        if len(operations) > 0:
            subprocess.check_call(["gcloud", "compute", "config-ssh"])
        time.sleep(20)
        self.init_node(self.controller_name())
        threads = []
        for worker in range(self.num_workers):
            process = threading.Thread(target=self.init_node, args=(self.worker_name(worker),))
            process.start()
            threads.append(process)
        for thread in threads:
            thread.join()
        self.remote_execute(self.controller_name(), "sudo systemctl restart slurmctld")

    def destroy(self):
        """Deletes the controller node and workers, including disks"""
        # self.remote_execute(
        #     self.controller_name(),
        #     "mkdir -p /mnt/cluster/results && sudo rsync -av /mnt/cluster/results/ /gs-bucket/results/",
        # )
        operations = []
        for instance in self.list_instances():
            name = instance["name"]
            print(f"Deleting node '{name}'")
            op = (
                self.compute.instances()
                .delete(project=self.project, zone=self.zone, instance=name)
                .execute()
            )
            operations.append(op)
        return operations

    def create_controller_node(self):
        print(f"Creating controller node '{self.controller_name()}'")
        config = {
            **self.base_config,
            "disks": [
                *self.base_config["disks"],
                {  # SSD for the shared file system /mnt/cluster
                    "mode": "READ_WRITE",
                    "autoDelete": True,
                    "deviceName": "local-ssd-0",
                    "type": "SCRATCH",
                    "interface": "NVME",
                    "initializeParams": {
                        "diskType": f"projects/{self.project}/zones/{self.zone}/diskTypes/local-ssd"
                    },
                },
            ],
            "name": self.controller_name(),
            "machineType": f"zones/{self.zone}/machineTypes/{self.controller_machine_type}",
        }
        return (
            self.compute.instances()
            .insert(project=self.project, zone=self.zone, body=config)
            .execute()
        )

    def create_worker_node(self, number):
        print(f"Creating worker node '{self.worker_name(number)}'")
        config = {
            **self.base_config,
            "name": self.worker_name(number),
            "machineType": f"zones/{self.zone}/machineTypes/{self.machine_type}",
            "guestAccelerators": [],
            "metadata": {"items": []},
        }

        if self.num_gpus_per_worker > 0:
            config["guestAccelerators"].append(
                {
                    "acceleratorCount": self.num_gpus_per_worker,
                    "acceleratorType": f"projects/{self.project}/zones/{self.zone}/acceleratorTypes/{self.gpu_type}",
                }
            )
            config["metadata"]["items"].append({"key": "install-nvidia-driver", "value": True})

        return (
            self.compute.instances()
            .insert(project=self.project, zone=self.zone, body=config)
            .execute()
        )

    def wait_to_finish(self, operations):
        """Block until all of the passed operation objects (returned by .execute()) are DONE."""
        if not isinstance(operations, list):
            operations = [operations]
        if len(operations) > 0:
            print("Waiting for operations to finish ...")
        for op in operations:
            while True:
                result = (
                    self.compute.zoneOperations()
                    .get(project=self.project, zone=self.zone, operation=op["name"])
                    .execute()
                )
                if result["status"] == "DONE":
                    if "error" in result:
                        raise Exception(result["error"])
                    break
                time.sleep(10)

    def init_node(self, hostname):
        print(f"[{hostname}] initializing")
        self.upload_file_contents(hostname, "slurm/slurm.conf", self.slurm_conf())
        self.upload_file_contents(hostname, "startup.sh", self.startup_script())
        self.upload_file_contents(hostname, "self_destruct.sh", self.self_destruct_script())

        # Update jobmonitor. Todo: move this to the snapshot once things are stable
        # wheel_filename = os.path.basename(self.jobmonitor_whl)
        # self.upload_file(hostname, f"install/{wheel_filename}", self.jobmonitor_whl)
        # self.remote_execute(
        #     hostname, f"/opt/anaconda3/bin/pip install --user --upgrade install/{wheel_filename}"
        # )

        self.remote_execute(hostname, "sudo bash ./startup.sh")

    def upload_file_contents(self, hostname, dest, file_content):
        f = tempfile.NamedTemporaryFile(delete=False, mode="w")
        f.write(file_content)
        f.close()
        self.upload_file(hostname, dest, f.name)
        os.unlink(f.name)

    def upload_file(self, hostname, dest, src):
        print(f"[{hostname}] uploading {dest}")
        fullname = f"{hostname}.{self.zone}.{self.project}"
        output = subprocess.check_output(
            ["scp", "-q", "-o", "StrictHostKeyChecking=no", src, f"{fullname}:{dest}"]
        )
        for line in output.splitlines(True):
            line = line.decode("utf-8")
            print(f"[{hostname}] {line}")

    def remote_execute(self, hostname, command):
        print(f"[{hostname}] {command}")
        fullname = f"{hostname}.{self.zone}.{self.project}"
        output = subprocess.check_output(
            ["ssh", "-o", "StrictHostKeyChecking=no", fullname, "--"] + command.split(" ")
        )
        for line in output.splitlines():
            line = line.decode("utf-8")
            print(f"[{hostname}] {line}")

    def slurm_conf(self):
        worker_range = self.worker_name("[0-{}]".format(self.num_workers - 1))

        assert self.machine_type.startswith("n1-standard-")
        cores_per_socket = int(self.machine_type.replace("n1-standard-", "")) // 2
        threads_per_core = 2

        return f"""
        SlurmctldHost={self.controller_name()}
        #SlurmctldHost=
        #
        #DisableRootJobs=NO
        #EnforcePartLimits=NO
        #Epilog=
        #EpilogSlurmctld=
        #FirstJobId=1
        #MaxJobId=999999
        GresTypes=gpu,mps
        #GroupUpdateForce=0
        #GroupUpdateTime=600
        #JobFileAppend=0
        #JobRequeue=1
        #JobSubmitPlugins=1
        #KillOnBadExit=0
        #LaunchType=launch/slurm
        #Licenses=foo*4,bar
        #MailProg=/bin/mail
        #MaxJobCount=5000
        #MaxStepCount=40000
        #MaxTasksPerNode=128
        MpiDefault=pmix
        #MpiParams=ports=#-#
        #PluginDir=
        #PlugStackConfig=
        #PrivateData=jobs
        ProctrackType=proctrack/cgroup
        #Prolog=
        #PrologFlags=
        #PrologSlurmctld=
        #PropagatePrioProcess=0
        #PropagateResourceLimits=
        #PropagateResourceLimitsExcept=
        #RebootProgram=
        ReturnToService=1
        #SallocDefaultCommand=
        SlurmctldPidFile=/var/run/slurmctld.pid
        SlurmctldPort=6817
        SlurmdPidFile=/var/run/slurmd.pid
        SlurmdPort=6818
        SlurmdSpoolDir=/var/spool/slurm/d
        SlurmUser=slurm
        #SlurmdUser=root
        #SrunEpilog=
        #SrunProlog=
        StateSaveLocation=/var/spool/slurm/ctld
        SwitchType=switch/none
        #TaskEpilog=
        TaskPlugin=task/cgroup
        TaskPluginParam=Sched
        #TaskProlog=
        #TopologyPlugin=topology/tree
        #TmpFS=/tmp
        #TrackWCKey=no
        #TreeWidth=
        #UnkillableStepProgram=
        #UsePAM=0
        #
        #
        # TIMERS
        #BatchStartTimeout=10
        #CompleteWait=0
        #EpilogMsgTime=2000
        #GetEnvTimeout=2
        #HealthCheckInterval=0
        #HealthCheckProgram=
        InactiveLimit=0
        KillWait=30
        #MessageTimeout=10
        #ResvOverRun=0
        MinJobAge=300
        #OverTimeLimit=0
        SlurmctldTimeout=120
        SlurmdTimeout=300
        #UnkillableStepTimeout=60
        #VSizeFactor=0
        Waittime=0
        #
        #
        # SCHEDULING
        #DefMemPerCPU=0
        #MaxMemPerCPU=0
        #SchedulerTimeSlice=30
        SchedulerType=sched/backfill
        SelectType=select/cons_tres
        SelectTypeParameters=CR_Core
        #
        #
        # JOB PRIORITY
        #PriorityFlags=
        #PriorityType=priority/basic
        #PriorityDecayHalfLife=
        #PriorityCalcPeriod=
        #PriorityFavorSmall=
        #PriorityMaxAge=
        #PriorityUsageResetPeriod=
        #PriorityWeightAge=
        #PriorityWeightFairshare=
        #PriorityWeightJobSize=
        #PriorityWeightPartition=
        #PriorityWeightQOS=
        #
        #
        # LOGGING AND ACCOUNTING
        #AccountingStorageEnforce=0
        #AccountingStorageHost=
        #AccountingStorageLoc=
        #AccountingStoragePass=
        #AccountingStoragePort=
        AccountingStorageType=accounting_storage/none
        #AccountingStorageUser=
        AccountingStoreJobComment=YES
        ClusterName=cluster
        #DebugFlags=
        #JobCompHost=
        #JobCompLoc=
        #JobCompPass=
        #JobCompPort=
        JobCompType=jobcomp/none
        #JobCompUser=
        #JobContainerType=job_container/none
        JobAcctGatherFrequency=30
        JobAcctGatherType=jobacct_gather/none
        SlurmctldDebug=info
        #SlurmctldLogFile=
        SlurmdDebug=info
        #SlurmdLogFile=
        #SlurmSchedLogFile=
        #SlurmSchedLogLevel=
        #
        #

        # CredType=cred/none
        # AuthType=auth/none

        # POWER SAVE SUPPORT FOR IDLE NODES (optional)
        #SuspendProgram=/etc/slurm/suspend.sh
        #ResumeProgram=/etc/slurm/resume.sh
        #SuspendTimeout=120
        #ResumeTimeout=120
        #ResumeRate=
        #SuspendExcNodes=
        #SuspendExcParts=
        #SuspendRate=
        #SuspendTime=60
        #
        #
        # COMPUTE NODES
        NodeName={worker_range} Gres=gpu:{self.gpu_type_slurm}:{self.num_gpus_per_worker},mps:{100*self.num_gpus_per_worker} Sockets=1 CoresPerSocket={cores_per_socket} ThreadsPerCore={threads_per_core} State=UNKNOWN
        PartitionName=low Nodes={worker_range} Default=NO MaxTime=INFINITE OverSubscribe=NO PriorityTier=10 PreemptMode=off State=UP
        PartitionName=med Nodes={worker_range} Default=YES MaxTime=INFINITE OverSubscribe=NO PriorityTier=20 PreemptMode=off State=UP
        PartitionName=high  Nodes={worker_range} Default=NO MaxTime=INFINITE OverSubscribe=NO PriorityTier=30 PreemptMode=off State=UP

        """.replace(
            "        ", ""
        )

    def startup_script(self):
        return f"""
        # Mount the Google Storage Bucket
        mkdir -p /gs-bucket
        sudo chmod 777 /gs-bucket
        gcsfuse -o allow_other -file-mode=777 -dir-mode=777 {self.bucket_name} /gs-bucket

        # Generate /etc/slurm/gres.conf
        GRES_CONF_FILE=/etc/slurm/gres.conf
        echo "AutoDetect=nvml" > $GRES_CONF_FILE
        COUNT="0"
        for i in `lspci | grep -i nvidia | grep -v Audio | awk '{{print $1}}' | cut -d : -f 1`
        do
            CPUAFFINITY=`cat /sys/class/pci_bus/0000:$i/cpulistaffinity`
            echo "Name=gpu Type={self.gpu_type_slurm} File=/dev/nvidia"$COUNT" Cores=$CPUAFFINITY" >> $GRES_CONF_FILE
            echo "Name=mps Count=100 File=/dev/nvidia"$COUNT >> $GRES_CONF_FILE
            ((++COUNT))
        done

        # Setup slurm configuration
        mv slurm/slurm.conf /etc/slurm/slurm.conf
        chown -R slurm:slurm /etc/slurm
        chmod 644 /etc/slurm/*.conf
        systemctl stop slurmctld
        rm -rf /var/spool/slurm/ctld/*  # remove old state

        # Start the right deamons
        if [ "$HOSTNAME" = {self.controller_name()} ]; then
            # systemctl restart slurmctld
            echo "We will start the slurm controller later"
        else
            systemctl restart slurmd
        fi

        # Setup the shared file system
        MOUNT_POINT=/mnt/cluster
        mkdir -p $MOUNT_POINT
        chmod -R 777 $MOUNT_POINT
        if [ "$HOSTNAME" = {self.controller_name()} ]; then
            LOCAL_SSD_DEVICE=/dev/nvme0n1
            mkfs.ext4 -F $LOCAL_SSD_DEVICE
            mount $LOCAL_SSD_DEVICE $MOUNT_POINT
            chmod -R 777 $MOUNT_POINT
            echo "$MOUNT_POINT *(rw,fsid=0,sync,no_root_squash,no_subtree_check)" > /etc/exports
            systemctl restart nfs-kernel-server
            systemctl start rpc-statd
        else
            echo "" > /etc/exports
            echo "{self.controller_name()}:$MOUNT_POINT $MOUNT_POINT nfs rw,intr,soft 0 0" >> /etc/fstab
            mount $MOUNT_POINT
        fi

        function sjobrun() {{
            jobid=${{1}}
            num_workers=`jobshow $jobid | grep n_workers | awk -F ' ' '{{print $2}}'`
            name=`jobshow $jobid | grep job | awk -F ' ' '{{print $2}}'`
            sbatch --ntasks $num_workers --gpus-per-task=1 --job-name="$name" --wrap="srun jobrun $jobid --mpi"
        }}
        """.replace(
            "        ", ""
        )

    def self_destruct_script(self):
        hosts = " ".join(
            [self.controller_name()] + [self.worker_name(w) for w in range(self.num_workers)]
        )
        return f"""
        #!/bin/bash

        # Stop on first error
        set -e

        gcloud compute instances delete {hosts} --quiet --zone {self.zone}
        """

    def file_server_setup_script(self):
        return f"""
        """.replace(
            "        ", ""
        )

    def list_instances(self):
        res = self.compute.instances().list(project=self.project, zone=self.zone).execute()
        if not "items" in res:
            return []
        instances = res["items"]
        return [i for i in instances if i["name"].startswith(self.name)]

    def controller_exists(self):
        res = self.compute.instances().list(project=self.project, zone=self.zone).execute()
        if not "items" in res:
            return False
        instances = res["items"]
        return any(i["name"] == self.controller_name() for i in instances)

    def worker_exists(self, number):
        res = self.compute.instances().list(project=self.project, zone=self.zone).execute()
        if not "items" in res:
            return False
        instances = res["items"]
        return any(i["name"] == self.worker_name(number) for i in instances)

    def worker_name(self, number):
        return f"{self.name}-worker-{number}"

    def controller_name(self):
        return self.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("action", default="create", choices=["create", "destroy", "exec"])

    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--num-workers", "-n", default=DEFAULT_NUM_WORKERS, type=int)
    parser.add_argument(
        "--num-gpus-per-worker", "-g", default=DEFAULT_NUM_GPUS_PER_WORKER, type=int
    )
    parser.add_argument("--zone", default=DEFAULT_ZONE, choices=["europe-west1-b", "us-central1-a"])
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--gpu-type", default=DEFAULT_GPU_TYPE, choices=["nvidia-tesla-k80", "nvidia-tesla-v100"]
    )
    machine_types = [
        "n1-standard-1",
        "n1-standard-2",
        "n1-standard-4",
        "n1-standard-8",
        "n1-standard-16",
        "n1-standard-32",
        "n1-standard-64",
    ]
    parser.add_argument("--machine-type", default=DEFAULT_MACHINE_TYPE, choices=machine_types)
    parser.add_argument(
        "--controller-machine-type", default=DEFAULT_CONTROLLER_MACHINE_TYPE, choices=machine_types
    )
    parser.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)
    parser.add_argument("--bucket-name", default=DEFAULT_BUCKET_NAME)

    parser.add_argument("--cmd", "-c", help="Command to execute (only for exec)")
    parser.add_argument("--preemptible", default=DEFAULT_PREEMPTIBLE, action="store_true")

    args = parser.parse_args()

    cluster = Cluster(
        name=args.name,
        num_workers=args.num_workers,
        project=args.project,
        zone=args.zone,
        image=args.image,
        machine_type=args.machine_type,
        gpu_type=args.gpu_type,
        num_gpus_per_worker=args.num_gpus_per_worker,
        controller_machine_type=args.controller_machine_type,
        service_account=args.service_account,
        bucket_name=args.bucket_name,
        is_preemptible=args.preemptible,
    )
    if args.action == "create":
        cluster.spin_up()
    elif args.action == "destroy":
        ops = cluster.destroy()
        cluster.wait_to_finish(ops)
    elif args.action == "exec":
        assert args.cmd is not None
        threads = []
        for worker in range(cluster.num_workers):
            process = threading.Thread(
                target=cluster.remote_execute, args=(cluster.worker_name(worker), args.cmd)
            )
            process.start()
            threads.append(process)
        for thread in threads:
            thread.join()
