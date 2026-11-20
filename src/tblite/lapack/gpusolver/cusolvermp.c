/* This file is part of tblite. SPDX-License-Identifier: LGPL-3.0-or-later */
#if TBLITE_HAS_CUSOLVERMP
#include <cuda_runtime_api.h>
#include <cusolverMp.h>
#include <limits.h>
#include <mpi.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { message_size = 256 };

struct mp_workspace {
    MPI_Comm comm;
    int device, n;
    size_t tile_bytes, device_bytes, host_bytes;
    cudaStream_t stream;
    ncclComm_t nccl;
    cusolverMpHandle_t solver;
    cusolverMpGrid_t grid;
    cusolverMpMatrixDescriptor_t desc;
    double *h, *s, *z, *eval;
    int *info;
    void *device_work, *host_work;
};

/* All ranks take the same exit before entering the next collective. */
static int sync_error(MPI_Comm comm, char msg[message_size])
{
    int rank, owner;
    MPI_Comm_rank(comm, &rank);
    owner = msg[0] ? rank : INT_MAX;
    MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MIN, comm);
    if (owner == INT_MAX) return 0;
    MPI_Bcast(msg, message_size, MPI_CHAR, owner, comm);
    return 1;
}

/* A stage keeps the first vendor error. Collective calls get their own stage. */
#define TRY(call) do { if (!msg[0]) { int rc = (call); if (rc) \
    snprintf(msg, message_size, "%s returned %d", #call, rc); } } while (0)

static int select_device(MPI_Comm comm, int rank, int *device, char *msg)
{
    MPI_Comm node;
    int local_rank, local_size, visible = 0;
    char pci[32] = {0};
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_size(node, &local_size);
    TRY(cudaGetDeviceCount(&visible));
    *device = visible == 1 ? 0 : local_rank;
    if (!msg[0] && *device >= visible)
        snprintf(msg, message_size, "local rank %d has only %d visible devices", local_rank, visible);
    TRY(cudaSetDevice(*device));
    TRY(cudaDeviceGetPCIBusId(pci, sizeof(pci), *device));
    char (*devices)[sizeof(pci)] = calloc((size_t)local_size, sizeof(pci));
    if (!devices) snprintf(msg, message_size, "cannot allocate the local device map");
    if (!sync_error(node, msg)) {
        MPI_Allgather(pci, sizeof(pci), MPI_CHAR, devices, sizeof(pci), MPI_CHAR, node);
        for (int i = 0; i < local_rank; ++i) {
            if (!strcmp(pci, devices[i])) {
                snprintf(msg, message_size, "local ranks %d and %d both selected GPU %s", i, local_rank, pci);
                break;
            }
        }
    }
    free(devices);
    MPI_Comm_free(&node);
    return sync_error(comm, msg);
}

void tblite_mp_delete(struct mp_workspace *w)
{
    if (!w) return;
    cudaSetDevice(w->device);
    if (w->stream) cudaStreamSynchronize(w->stream);
    if (w->device_work) cudaFree(w->device_work);
    if (w->host_work) cudaFreeHost(w->host_work);
    if (w->info) cudaFree(w->info);
    if (w->eval) cudaFree(w->eval);
    if (w->z) cudaFree(w->z);
    if (w->s) cudaFree(w->s);
    if (w->h) cudaFree(w->h);
    if (w->desc) cusolverMpDestroyMatrixDesc(w->desc);
    if (w->grid) cusolverMpDestroyGrid(w->grid);
    if (w->solver) cusolverMpDestroy(w->solver);
    if (w->nccl) ncclCommDestroy(w->nccl);
    if (w->stream) cudaStreamDestroy(w->stream);
    free(w);
}

int tblite_mp_create(struct mp_workspace **out, int fcomm, int n, int nb, int npcol,
                     int lld, int local_cols, char *msg)
{
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)fcomm);
    ncclUniqueId uid;
    int rank, size, device;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    *out = NULL;
    msg[0] = '\0';
    if (select_device(comm, rank, &device, msg)) return 1;
    struct mp_workspace *w = calloc(1, sizeof(*w));
    if (!w) snprintf(msg, message_size, "cannot allocate the solver handle");
    if (sync_error(comm, msg)) { free(w); return 1; }
    w->comm = comm;
    w->device = device;
    w->n = n;
    size_t cols = local_cols > 0 ? (size_t)local_cols : 1;
    if (lld < 1 || cols > SIZE_MAX / sizeof(double) / (size_t)lld)
        snprintf(msg, message_size, "invalid local matrix dimensions");
    w->tile_bytes = (size_t)lld * cols * sizeof(double);
    TRY(cudaStreamCreate(&w->stream));
    if (rank == 0) TRY(ncclGetUniqueId(&uid));
    if (sync_error(comm, msg)) goto fail;
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, comm);
    TRY(ncclCommInitRank(&w->nccl, size, uid, rank));
    if (sync_error(comm, msg)) goto fail;
    TRY(cusolverMpCreate(&w->solver, device, w->stream));
    if (sync_error(comm, msg)) goto fail;
    TRY(cusolverMpCreateDeviceGrid(w->solver, &w->grid, w->nccl, size / npcol, npcol,
                                  CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));
    if (sync_error(comm, msg)) goto fail;
    TRY(cusolverMpCreateMatrixDesc(&w->desc, w->grid, CUDA_R_64F, n, n, nb, nb, 0, 0, lld));
    if (sync_error(comm, msg)) goto fail;
    TRY(cusolverMpSygvd_bufferSize(w->solver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n, 1, 1, w->desc, 1, 1, w->desc, 1, 1, w->desc, CUDA_R_64F, &w->device_bytes, &w->host_bytes));
    TRY(cudaMalloc((void **)&w->h, w->tile_bytes));
    TRY(cudaMalloc((void **)&w->s, w->tile_bytes));
    TRY(cudaMalloc((void **)&w->z, w->tile_bytes));
    TRY(cudaMalloc((void **)&w->eval, (size_t)n * sizeof(double)));
    TRY(cudaMalloc((void **)&w->info, sizeof(int)));
    if (w->device_bytes) TRY(cudaMalloc(&w->device_work, w->device_bytes));
    if (w->host_bytes) TRY(cudaMallocHost(&w->host_work, w->host_bytes));
    if (sync_error(comm, msg)) goto fail;
    *out = w;
    return 0;
fail:
    tblite_mp_delete(w);
    return 1;
}

int tblite_mp_solve(struct mp_workspace *w, const double *h, const double *s,
                    double *eval, double *z, char *msg)
{
    int info = -1;
    msg[0] = '\0';
    TRY(cudaSetDevice(w->device));
    TRY(cudaMemcpyAsync(w->h, h, w->tile_bytes, cudaMemcpyHostToDevice, w->stream));
    TRY(cudaMemcpyAsync(w->s, s, w->tile_bytes, cudaMemcpyHostToDevice, w->stream));
    TRY(cudaStreamSynchronize(w->stream));
    if (sync_error(w->comm, msg)) return 1;
    TRY(cusolverMpSygvd(w->solver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        w->n, w->h, 1, 1, w->desc, w->s, 1, 1, w->desc, w->eval, w->z, 1, 1, w->desc,
        CUDA_R_64F, w->device_work, w->device_bytes, w->host_work, w->host_bytes, w->info));
    TRY(cudaMemcpyAsync(&info, w->info, sizeof(info), cudaMemcpyDeviceToHost, w->stream));
    TRY(cudaMemcpyAsync(eval, w->eval, (size_t)w->n * sizeof(double), cudaMemcpyDeviceToHost, w->stream));
    TRY(cudaMemcpyAsync(z, w->z, w->tile_bytes, cudaMemcpyDeviceToHost, w->stream));
    TRY(cudaStreamSynchronize(w->stream));
    if (!msg[0] && info != 0) snprintf(msg, message_size, "cusolverMpSygvd device info = %d", info);
    return sync_error(w->comm, msg);
}
#endif
