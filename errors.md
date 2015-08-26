From: https://github.com/BVLC/caffe/issues/2720 

Answer from xinyanhe 

```
dyld: Library not loaded: libhdf5_hl.10.dylib
```
I directly downloaded the source code of 1.8.15 version from HDF website: http://www.hdfgroup.org/HDF5/release/obtainsrc.html#src


Go to:  `Distributions containing Unix line endings:`   and download the  `[ gzip ]`

After downloaded, follow the instructions here: http://www.hdfgroup.org/ftp/HDF5/current/src/unpacked/release_docs/INSTALL



We can see that there is already a folder called '1.8.14' under `/usr/local/Cellar/hdf5` if you have installed the hdf5 through brew.

Clone the folder /usr/local/hdf5 into the /usr/local/Cellar/hdf5, and rename it to1.8.15.
run brew switch hdf5 1.8.15 to make brew switch to the specific version.
And now I can run make runtest successfully.
```
$ /usr/local/Cellar/hdf5|
```

```
$ brew switch hdf5 1.8.15
```

Cleaning /usr/local/Cellar/hdf5/1.8.14
Cleaning /usr/local/Cellar/hdf5/1.8.15
69 links created for /usr/local/Cellar/hdf5/1.8.15