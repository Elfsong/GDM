sudo mdadm --create /dev/md0 --level=0 --raid-devices=2  /dev/disk/by-id/google-local-nvme-ssd-0  /dev/disk/by-id/google-local-nvme-ssd-1
sudo mkfs.ext4 -F /dev/md0
sudo mount /dev/md0 /mnt/data/
sudo chmod a+w /mnt/data/