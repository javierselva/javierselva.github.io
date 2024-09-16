---
layout: post
section-type: post
has-comments: false
title: "Checking the state of your HDD with badblocks"
category: tnt
tags: ["tnt","hdd","drive","check","badblocks"]
---

# Introduction
Since I had an hdd scare and almost lost all of my childhood pictures I started taking backups and disk status seriously. There are many ways to do this. You can pay for cloud storage and let professionals handle it. Or if you're feeling techie you can set up some raid build so your data replicates automatically. But if like me you're on a budget, the easiest way I've found is to manually hold several copies in different hard drives, and ideally have them in different places.

But that's not all, you want to regularly check the state of your hard drives. In principle they should last you for a long time, but you can even find new drives to have some bad blocks. And this will only get worse with time and the more you use them. There are some utilities like [CrystalDiskInfo](https://en.wikipedia.org/wiki/CrystalDiskMark) for Windows or [`smartctl`](https://www.smartmontools.org/) for Linux that allow you to get a good estimate on the state of your drive. However, these tools use a technology known as S.M.A.R.T. (Self-Monitoring, Analysis, and Reporting Technology) which pasively reports cases of bad blocks in hard drvies. For most day to day cases this is enough, they can let you know if something starts to look badly. If these quick tests ([see below](#s.m.a.r.t.)) already report some problems, the safest option is to just back up the drive before doing anything else. If it is in very bad shape you may even need [specialized tools to do so](https://askubuntu.com/questions/313081/what-tool-should-i-use-for-a-bit-by-bit-copy-of-my-hard-drive). However, modern drives supporting S.M.A.R.T. are capable of handling a few bad blocks, so you may want to assess the magnitude of the problem with some other tool. Similarly, for new drives you plan to use as back up, it may be wise to run some deeper test, even once a year. 

# Badblocks
To do this, you can use [`badblocks`](https://wiki.archlinux.org/title/Badblocks). This tool will read/write from/to every single block in your hard drive. Badblocks is intense, as it will perform millions of read-write operations, stressing the (potentially already damaged) hard drive. Ideally, badblocks should report 0 bad blocks on a new device. If you have information you wish to keep in the drive, you should perform a **read only test**, as a read/write test will entirely remove all data contained in the disk. In any case, before doing anything, **always back-up your data**.

To use it you need to follow these steps:

1. Install `badblocks`
```bash
sudo apt-get install e2fsprogs
```
Now you're ready to run the test. Bear in mind this may take several **days** depending on the size of your drive. It will perform the test several times to make absolutely sure everything is fine.

2. Find out your hard drive address:
```bash
sudo lshw -class disk
```
This will give out something like:
```
*-disk                    
       description: SCSI Disk
       product: 541010A9E680
       vendor: HGST HTS
       physical id: 0.0.0
       bus info: scsi@2:0.0.0
       logical name: /dev/sdb
       size: 931GiB (1TB)
       capabilities: gpt-1.00 partitioned partitioned:gpt
       configuration: ansiversion=6 guid=88003064-c468-484a-852f-8b089508b84e logicalsectorsize=512 sectorsize=512
  *-disk
       description: ATA Disk
       product: Intenso SSD Sata
       physical id: 0
       bus info: scsi@0:0.0.0
       logical name: /dev/sda
       version: 9A0
       serial: AYSAU1227OUT000094
       size: 953GiB (1024GB)
       capabilities: gpt-1.00 partitioned partitioned:gpt
       configuration: ansiversion=5 guid=1589a582-3c6d-4587-8488-532feb1e6e9c logicalsectorsize=512 sectorsize=512
```
Whith the information provided it should be enough to identify the disk. If you still can't tell which one it is, you may want to try [`gparted`](https://gparted.org/), with the GUI showing partitions it may be easier to tell apart the disks. In my case, the SSD is my internal hard drive, and the other disk is the external one I need to check, which logical name is `/dev/sdb` (for all commands below make sure you replace this with the name of your disk).

3. You may need to unmount the device before badblocks can analyze your device:
```bash
lsblk /dev/sdb
```
Which will give you somehting like:
```
NAME   MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
sdb      8:16   0 931,5G  0 disk 
├─sdb1   8:17   0   549G  0 part /media/username/Windows10
```
So then you can run:
```bash
umount /media/username/Windows10
```

4. Actually run the test.
    * A) Run a destructive test. This will rewrite everything in your drive, so a backup of your data is highly recommended. This test makes the most sense on a new or empty drive. I'm using the `-b 4096` option, which specifies the block size, and is supposed to make the run very slightly faster.
    ```bash
    sudo badblocks -wsv -b 4096 /dev/sdb
    ```
    This will write four different patterns on each individual block, then read them to check everything is fine. Effectively, **this will remove all data in the disk**.

    * B) You can alternatively perform a non-destructive test.
    ```bash
    sudo badblocks -nsv -b 4096 /dev/sdb
    ```
    This will read the contents of each block, perform the test, and then write the original contents back. In principle this should leave all your data untouched, however, may there be any bad blocks in your drive, it is always safer to perform a backup of your data before running the test.

    * C) If you want to run without danger of damaging your data, you can simply run:
    ```bash
    sudo badblocks -sv -b 4096 /dev/sdb
    ```
    Which will perform the read-only test, ensuring no data is rewriten or broken in the process, but it is not as informative on the health of the disk. However, given that badblocks is an old tool originally written for floppy disks, doing the read-only test that `smartctl` provides may be slightly faster:
    ```bash
    smartctl -t long /dev/sdb
    ```
    This test you can afford to perform regularly, even on larger disks, as it won't take several days of your time.

5. Let's take a loot at the results. Badblocks output looks like this:
```
Checking for bad blocks in read-write mode
From block 0 to 244190645
Testing with pattern 0xaa: done                                                 
Reading and comparing: done   
Testing with pattern 0x55: done                                                 
Reading and comparing: done  
Testing with pattern 0xff: done                                                 
Reading and comparing: done  
Testing with pattern 0x00: done                                                 
Reading and comparing: done                                                
Pass completed, 0 bad blocks found. (0/0/0 errors)
```
The errors are reported as read/write/compare. The two former errors are reported by the disk when performing a specific read/write operation. The compare one is reported by badblocks when the written and read data is different, meaning the disk did not report any problems when reading/writing but the data was corrupted at some point. Note that these errors only appear when using `-w` or `-n`, and never during a read-only test (nor will the writting errors).

Extra. I was having some trouble running the whole test. I'm not sure whether it was the drive, my computer or badblocks, but either the drive ended up disconnecting (getting "invalid argument during seek") or the PC would hang. To make sure I could run the test completely, I decided to manually run the test with each of the four patterns badblock uses (0xaa, 0x55, 0xff, 0x00), so I could run one, give the drive a rest, and then another. To manually specify the pattern you can use the `-t` option. You can also specify "random" instead of one of the four patterns above. Nevertheless, those four patterns are thought to be executed in that particular order so all bits in the disk hold a 1 and a 0 at some point, and all of them transition from 1 -> 0 and viceversa.
```bash
    sudo badblocks -wsv -t 0xaa -b 4096 /dev/sdb
```


# S.M.A.R.T.
Now, does a bad sector mean I need to change the disk? Yes. And no. Some bad sectors could indicate the beginning of the demise of your drive, and **you should inmediately back up the data** contained on it, some people suggest to absolutely stop using the device after that. However, modern drives using S.M.A.R.T. are capable of reallocating those sectors and make sure they're never used. So, if you find any errors during the badblocks run, I strongly recommend checking the device with `smartctl`. This technology automatically runs some tests on the health of the device, and is able to report that in [multiple ways](https://www.thomas-krenn.com/en/wiki/SMART_tests_with_smartctl). You can use this as a complement to the deeper badblocks' test, as it will provide with more informative health status of the disk, beyond blocks.

First, you need to install it with
```bash
sudo apt-get install smartmontools
```
For a short summary:
```bash
smartctl --health /dev/sdb
```
Or, as I mentioned above, you can run some tests yourself, either short or long, the main difference being that the short will only test one sector of the device, whereas the long one tests the whole disk. Remember this is a read-only test, so if the device is in good shape it should not affect the data stored on the device.

```bash
smartctl -t <short|long> /dev/sdb
```
For a slightly longer report, and to see the results of the tests above, you can do the following. Also, here's [a handy guide](https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology#ATA_S.M.A.R.T._attributes) to try and understand the output of this command.
```bash
smartctl --all /dev/sdb
```
Here's some example of the output of this last command. First, It will show general information about the device and a general status grade, in this case, "PASSED", meaning everything is looking good overall.

```
=== START OF INFORMATION SECTION ===
Model Family:     Seagate Samsung SpinPoint M8 (AF)
Device Model:     ST1000LM024 HN-M101MBB
Serial Number:    S30YJ9JFB13101
LU WWN Device Id: 5 0004cf 20e8ded45
Firmware Version: 2BA30001
User Capacity:    1.000.204.886.016 bytes [1,00 TB]
Sector Sizes:     512 bytes logical, 4096 bytes physical
Rotation Rate:    5400 rpm
Form Factor:      2.5 inches
Device is:        In smartctl database [for details use: -P show]
ATA Version is:   ATA8-ACS T13/1699-D revision 6
SATA Version is:  SATA 3.0, 6.0 Gb/s (current: 3.0 Gb/s)
Local Time is:    Tue Sep  3 15:31:36 2024 CEST
SMART support is: Available - device has SMART capability.
SMART support is: Enabled

=== START OF READ SMART DATA SECTION ===
SMART Status not supported: Incomplete response, ATA output registers missing
SMART overall-health self-assessment test result: PASSED
Warning: This result is based on an Attribute check.
```

Now, if you scroll down a bit, you'll get to see **the attributes**:

```
SMART Attributes Data Structure revision number: 16
Vendor Specific SMART Attributes with Thresholds:
ID# ATTRIBUTE_NAME          FLAG     VALUE WORST THRESH TYPE      UPDATED  WHEN_FAILED RAW_VALUE
  1 Raw_Read_Error_Rate     0x002f   100   100   051    Pre-fail  Always       -       77
  2 Throughput_Performance  0x0026   252   252   000    Old_age   Always       -       0
  3 Spin_Up_Time            0x0023   092   088   025    Pre-fail  Always       -       2507
  4 Start_Stop_Count        0x0032   093   093   000    Old_age   Always       -       7932
  5 Reallocated_Sector_Ct   0x0033   252   252   010    Pre-fail  Always       -       0
  7 Seek_Error_Rate         0x002e   252   252   051    Old_age   Always       -       0
  8 Seek_Time_Performance   0x0024   252   252   015    Old_age   Offline      -       0
  9 Power_On_Hours          0x0032   100   100   000    Old_age   Always       -       10054
 10 Spin_Retry_Count        0x0032   252   252   051    Old_age   Always       -       0
 11 Calibration_Retry_Count 0x0032   100   100   000    Old_age   Always       -       812
 12 Power_Cycle_Count       0x0032   093   093   000    Old_age   Always       -       7942
191 G-Sense_Error_Rate      0x0022   100   100   000    Old_age   Always       -       1877
192 Power-Off_Retract_Count 0x0022   100   100   000    Old_age   Always       -       80
194 Temperature_Celsius     0x0002   061   054   000    Old_age   Always       -       39 (Min/Max 11/47)
195 Hardware_ECC_Recovered  0x003a   100   100   000    Old_age   Always       -       0
196 Reallocated_Event_Count 0x0032   252   252   000    Old_age   Always       -       0
197 Current_Pending_Sector  0x0032   252   100   000    Old_age   Always       -       0
198 Offline_Uncorrectable   0x0030   252   252   000    Old_age   Offline      -       0
199 UDMA_CRC_Error_Count    0x0036   200   200   000    Old_age   Always       -       0
200 Multi_Zone_Error_Rate   0x002a   100   100   000    Old_age   Always       -       18856
223 Load_Retry_Count        0x0032   100   100   000    Old_age   Always       -       812
225 Load_Cycle_Count        0x0032   051   051   000    Old_age   Always       -       496059
```
A couple of comments about this, because the first time I saw "pre-fail" and "old_age" I really freaked out. It seems like those are simply the type of attribute. To me this implies that, were that attribute to attain a lower value than the threshold, it would indicate that there is a "pre-fail" about to happen, or that the drive is old and should be replaced. In this case no value is lower than the threshold, so there's nothing to worry about. Also, the "VALUE" column uses some sort of renormalization to define a *score*, higher is always better and in most hard drives the maximum value is 253 (although some use 200 or 100). Note that "WORST" and "THRESHOLD" columns work within the same range. If you want to check the real value you need to look at the last column. For instance, there's been 0 *Reallocated_Sector_Ct*, which results in a max score of 252.