# Installation Instructions

## Install PDC and its dependent libraries

Please follow the instructions in the [PDC documentation website](https://pdc.readthedocs.io/en/latest/getting_started.html#installing-pdc-from-source-code) to install libfabric, mercury, and PDC. We recommend using the PDC stable branch.


## Environmental variables
The following  environment variables should be set before continuing.
```
export PDC_DIR="where/you/install/your/pdc"
export VOL_DIR="where/you/install/your/vol-pdc"
export HDF5_DIR="where/you/install/your/hdf5"
export MERCURY_DIR="where/you/install/your/mercury"
export LIBFABRIC_DIR="where/you/install/your/libfabric"
export LD_LIBRARY_PATH="$LIBFABRIC_DIR/lib:$MERCURY_DIR/lib:$LD_LIBRARY_PATH"
export PATH="$LIBFABRIC_DIR/include:$LIBFABRIC_DIR/lib:$MERCURY_DIR/include:$MERCURY_DIR/lib:$PATH"
```

## Install HDF5
```
git clone https://github.com/HDFGroup/hdf5.git
cd hdf5
git checkout hdf5_1_14_5
cmake -DCMAKE_INSTALL_PREFIX=$HDF5_DIR -DHDF5_ENABLE_PARALLEL=ON ../ 
make && make install
```

## Building vol-pdc
```
git clone https://github.com/hpc-io/vol-pdc.git
cd vol-pdc
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=$VOL_DIR -DBUILD_SHARED_LIBS=ON -DHDF5_DIR=$HDF5_DIR -DPDC_DIR=$PDC_DIR/share/cmake/pdc -DBUILD_EXAMPLES=ON
```

## Using vol-pdc
To utilize PDC via the PDC VOL connector, the following environment variables should be set:

```
export HDF5_PLUGIN_PATH=$VOL_DIR/lib
export HDF5_VOL_CONNECTOR="pdc under_vol=0;under_info={}"
export LD_LIBRARY_PATH="$LIBFABRIC_DIR/lib:$MERCURY_DIR/lib:$PDC_DIR/lib:$VOL_DIR/lib:$LD_LIBRARY_PATH"
# Optional
export LD_PRELOAD=$VOL_DIR/install/lib/libhdf5_vol_pdc.so
```

The connector calls `PDCinit`, which is collective over `MPI_COMM_WORLD`, at the first HDF5 call. If ranks do not make their first HDF5 call together (for example, only rank 0 creates a file), the application hangs. To avoid this, either call `H5open()` on every rank right after `MPI_Init`, or preload the helper that does it for you:

```
export LD_PRELOAD=$VOL_DIR/lib/libhdf5_vol_pdc_preinit.so
```

PDC servers must be launched before the application, more details can be found in the [PDC documentation website](https://pdc.readthedocs.io/en/latest/getting_started.html#running-pdc).


# Notes

The following functions have yet to be implemented and either currently do nothing, or don't do anything relevant to the VOL:

- H5VL_pdc_attr_get
- H5VL_pdc_attr_close
- H5VL_pdc_group_close
- H5VL_pdc_group_get
- H5VL_pdc_introspect_opt_query

The following functions have been modified to work in the context of the VOL, but contain extraneous code that is either never called by the VOL or isn't relevant to the VOL:
- H5VL_pdc_dataset_get
- H5VL_pdc_file_specific

