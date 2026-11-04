/*
 * Optional LD_PRELOAD helper for the PDC VOL connector.
 *
 * HDF5 loads the connector at the first HDF5 call, and the connector then calls
 * PDCinit, which is collective over MPI_COMM_WORLD. If one rank reaches HDF5 before
 * the others, for example because only rank 0 creates a file, that rank waits in
 * PDCinit for ranks that are not in a matching call, and the application hangs.
 * The connector cannot avoid this: the other ranks are outside its reach.
 *
 * This library calls H5open at the end of MPI_Init, while every rank is together,
 * so the connector initializes collectively. It is not needed when an application
 * calls H5open after MPI_Init itself or when its first HDF5 call is collective.
 *
 * PDC servers must be running before the application starts.
 */
#include <mpi.h>
#include <hdf5.h>

int
MPI_Init(int *argc, char ***argv)
{
    int ret = PMPI_Init(argc, argv);

    if (ret == MPI_SUCCESS)
        H5open();
    return ret;
}

int
MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
    int ret = PMPI_Init_thread(argc, argv, required, provided);

    if (ret == MPI_SUCCESS)
        H5open();
    return ret;
}
