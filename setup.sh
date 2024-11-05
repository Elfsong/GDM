sudo mdadm --create /dev/md0 --level=0 --raid-devices=2  /dev/disk/by-id/google-local-nvme-ssd-0  /dev/disk/by-id/google-local-nvme-ssd-1
sudo mkfs.ext4 -F /dev/md0
sudo mount /dev/md0 /mnt/data/
sudo chmod a+w /mnt/data/

echo "Cache Dir Mounted [/mnt/data/]"

sudo localedef -i en_US -f UTF-8 en_US.UTF-8