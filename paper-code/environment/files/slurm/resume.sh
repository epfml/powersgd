#!/bin/bash
echo "`date` Resume invoked $0 $*" >> /var/log/power_save.log
hosts=`scontrol show hostnames $1`
zone=$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)
for host in $hosts
do
   gcloud compute instances start $host --zone $zone --quiet
done
