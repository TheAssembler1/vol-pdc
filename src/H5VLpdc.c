/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the files COPYING and Copyright.html.  COPYING can be found at the root   *
 * of the source code distribution tree; Copyright.html can be found at the  *
 * root level of an installed copy of the electronic HDF5 document set and   *
 * is linked from the top-level documents page.  It can also be found at     *
 * http://hdfgroup.org/HDF5/doc/Copyright.html.  If you do not have          *
 * access to either file, you may request a copy from help@hdfgroup.org.     *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Purpose: The PDC VOL connector utilize the PDC library as the storage backend
 */
#include "H5VLpdc.h" /* PDC plugin                         */

#include "hdf5.h"
#include "H5PLextern.h"
#include "H5VLerror.h"

/* External headers needed by this file */
#include "H5linkedlist.h"
#include "pdc.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/****************/
/* Local Macros */
/****************/

#define ADDR_MAX              256
#define H5VL_PDC_SEQ_LIST_LEN 128

#ifdef PDC_VOL_WRITE_CACHE_MAX_GB
#define MAX_WRITE_CACHE_SIZE_GB PDC_VOL_WRITE_CACHE_MAX_GB
#else
#define MAX_WRITE_CACHE_SIZE_GB 1
#endif
/* (Uncomment to enable) */
/* #define ENABLE_LOGGING */

/*******************/
/* Global variables*/
/*******************/
uint64_t write_cache_size_g = 0;

/* Remove warnings when connector does not use callback arguments */
#if defined(__cplusplus)
#define H5VL_ATTR_UNUSED
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#define H5VL_ATTR_UNUSED __attribute__((unused))
#else
#define H5VL_ATTR_UNUSED
#endif

/************************************/
/* Local Type and Struct Definition */
/************************************/

/* The VOL wrapper context */
typedef struct H5VL_pdc_wrap_ctx_t {
    hid_t under_vol_id;   /* VOL ID for under VOL */
    void *under_wrap_ctx; /* Object wrapping context for under VOL */
} H5VL_pdc_wrap_ctx_t;

/* Common object information */
typedef struct H5VL_pdc_obj_t {
    hid_t          under_vol_id;
    void *         under_object;
    pdcid_t        obj_id;
    int            obj_type;
    char *         file_name;
    char           obj_name[ADDR_MAX];
    char *         group_name;
    char *         attr_name;
    psize_t        attr_value_size;
    pdc_var_type_t pdc_type;
    psize_t        compound_size;
    pdcid_t        reg_id_from;
    pdcid_t        reg_id_to;
    pdcid_t *      xfer_requests;
    int            req_alloc;
    int            req_cnt;
    H5I_type_t     h5i_type;
    H5O_type_t     h5o_type;
    void **        bufs;
    /* File object elements */
    MPI_Comm               comm;
    MPI_Info               info;
    int                    my_rank;
    int                    num_procs;
    pdcid_t                cont_id;
    int                    nobj;
    struct H5VL_pdc_obj_t *file_obj_ptr;
    H5_LIST_HEAD(H5VL_pdc_obj_t) ids;
    /* Dataset object elements */
    hid_t   dcpl_id;
    hid_t   dapl_id;
    hid_t   dxpl_id;
    hid_t   type_id;
    hid_t   space_id;
    hbool_t mapped;
    H5_LIST_ENTRY(H5VL_pdc_obj_t) entry;
} H5VL_pdc_obj_t;

/* PDC-specific file access properties */
typedef struct H5VL_pdc_info_t {
    void *under_vol_info;
    hid_t under_vol_id;
} H5VL_pdc_info_t;

/********************/
/* Local Prototypes */
/********************/

/* "Management" callbacks */
static herr_t H5VL_pdc_init(hid_t vipl_id);
static herr_t H5VL_pdc_obj_term(void);

/* VOL info callbacks */
static void * H5VL_pdc_info_copy(const void *_old_info);
static herr_t H5VL_pdc_info_cmp(int *cmp_value, const void *_info1, const void *_info2);
static herr_t H5VL_pdc_info_free(void *info);
static herr_t H5VL_pdc_info_to_str(const void *info, char **str);
static herr_t H5VL_pdc_str_to_info(const char *str, void **info);

/* VOL object wrap / retrieval callbacks */
static void * H5VL_pdc_get_object(const void *obj);
static herr_t H5VL_pdc_get_wrap_ctx(const void *obj, void **wrap_ctx);
static void * H5VL_pdc_wrap_object(void *obj, H5I_type_t obj_type, void *wrap_ctx);
static void * H5VL_pdc_unwrap_object(void *obj);
static herr_t H5VL_pdc_free_wrap_ctx(void *obj);

/* File callbacks */
static void * H5VL_pdc_file_create(const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id,
                                   hid_t dxpl_id, void **req);
static void * H5VL_pdc_file_open(const char *name, unsigned flags, hid_t fapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_file_get(void *file, H5VL_file_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_file_close(void *file, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_file_specific(void *file, H5VL_file_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_file_optional(void *file, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Dataset callbacks */
static void * H5VL_pdc_dataset_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                      hid_t lcpl_id, hid_t type_id, hid_t space_id, hid_t dcpl_id,
                                      hid_t dapl_id, hid_t dxpl_id, void **req);
static void * H5VL_pdc_dataset_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                    hid_t dapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_dataset_read(size_t count, void *dset[], hid_t mem_type_id[], hid_t mem_space_id[],
                                    hid_t file_space_id[], hid_t plist_id, void *buf[], void **req);
static herr_t H5VL_pdc_dataset_write(size_t count, void *dset[], hid_t mem_type_id[], hid_t mem_space_id[],
                                     hid_t file_space_id[], hid_t plist_id, const void *buf[], void **req);
static herr_t H5VL_pdc_dataset_get(void *dset, H5VL_dataset_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_dataset_specific(void *obj, H5VL_dataset_specific_args_t *args, hid_t dxpl_id,
                                        void **req);
static herr_t H5VL_pdc_dataset_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_dataset_close(void *dset, hid_t dxpl_id, void **req);

/* Datatype callbacks */
static void * H5VL_pdc_datatype_commit(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                       hid_t type_id, hid_t lcpl_id, hid_t tcpl_id, hid_t tapl_id,
                                       hid_t dxpl_id, void **req);
static void * H5VL_pdc_datatype_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                     hid_t tapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_datatype_get(void *dt, H5VL_datatype_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_datatype_specific(void *obj, H5VL_datatype_specific_args_t *args, hid_t dxpl_id,
                                         void **req);
static herr_t H5VL_pdc_datatype_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_datatype_close(void *dt, hid_t dxpl_id, void **req);

/* Attribute callbacks */
static void * H5VL_pdc_attr_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                   hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id, hid_t dxpl_id,
                                   void **req);
static void * H5VL_pdc_attr_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                 hid_t aapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_read(void *attr, hid_t mem_type_id, void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_write(void *attr, hid_t mem_type_id, const void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_get(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_specific(void *obj, const H5VL_loc_params_t *loc_params,
                                     H5VL_attr_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_attr_close(void *attr, hid_t dxpl_id, void **req);

/* Container/connector introspection callbacks */
static herr_t H5VL_pdc_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl,
                                               const H5VL_class_t **conn_cls);
static herr_t H5VL_pdc_introspect_get_cap_flags(const void *info, uint64_t *cap_flags);
static herr_t H5VL_pdc_introspect_opt_query(void *obj, H5VL_subclass_t cls, int opt_type, uint64_t *flags);

/* Group callbacks */
static void * H5VL_pdc_group_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                    hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id, hid_t dxpl_id, void **req);
static void * H5VL_pdc_group_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                                  hid_t gapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_group_get(void *obj, H5VL_group_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_group_specific(void *obj, H5VL_group_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_group_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_group_close(void *grp, hid_t dxpl_id, void **req);

/* Link callbacks */
static herr_t H5VL_pdc_link_create(H5VL_link_create_args_t *args, void *obj,
                                   const H5VL_loc_params_t *loc_params, hid_t lcpl_id, hid_t lapl_id,
                                   hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_link_copy(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj,
                                 const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id,
                                 hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_link_move(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj,
                                 const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id,
                                 hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_link_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_get_args_t *args,
                                hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_link_specific(void *obj, const H5VL_loc_params_t *loc_params,
                                     H5VL_link_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_link_optional(void *obj, const H5VL_loc_params_t *loc_params,
                                     H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Object callbacks */
static void * H5VL_pdc_object_open(void *obj, const H5VL_loc_params_t *loc_params, H5I_type_t *opened_type,
                                   hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_object_copy(void *src_obj, const H5VL_loc_params_t *src_loc_params,
                                   const char *src_name, void *dst_obj,
                                   const H5VL_loc_params_t *dst_loc_params, const char *dst_name,
                                   hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_object_get(void *obj, const H5VL_loc_params_t *loc_params,
                                  H5VL_object_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_object_specific(void *obj, const H5VL_loc_params_t *loc_params,
                                       H5VL_object_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pdc_object_optional(void *obj, const H5VL_loc_params_t *loc_params,
                                       H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Async request callbacks */
static herr_t H5VL_pdc_request_wait(void *req, uint64_t timeout, H5VL_request_status_t *status);
static herr_t H5VL_pdc_request_notify(void *obj, H5VL_request_notify_t cb, void *ctx);
static herr_t H5VL_pdc_request_cancel(void *req, H5VL_request_status_t *status);
static herr_t H5VL_pdc_request_specific(void *req, H5VL_request_specific_args_t *args);
static herr_t H5VL_pdc_request_optional(void *req, H5VL_optional_args_t *args);
static herr_t H5VL_pdc_request_free(void *req);

/* Blob callbacks */
static herr_t H5VL_pdc_blob_put(void *obj, const void *buf, size_t size, void *blob_id, void *ctx);
static herr_t H5VL_pdc_blob_get(void *obj, const void *blob_id, void *buf, size_t size, void *ctx);
static herr_t H5VL_pdc_blob_specific(void *obj, void *blob_id, H5VL_blob_specific_args_t *args);
static herr_t H5VL_pdc_blob_optional(void *obj, void *blob_id, H5VL_optional_args_t *args);

/* Token callbacks */
static herr_t H5VL_pdc_token_cmp(void *obj, const H5O_token_t *token1, const H5O_token_t *token2,
                                 int *cmp_value);
static herr_t H5VL_pdc_token_to_str(void *obj, H5I_type_t obj_type, const H5O_token_t *token,
                                    char **token_str);
static herr_t H5VL_pdc_token_from_str(void *obj, H5I_type_t obj_type, const char *token_str,
                                      H5O_token_t *token);

/* Generic optional callback */
static herr_t H5VL_pdc_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/*******************/
/* Local variables */
/*******************/

/* The PDC VOL plugin struct */
static const H5VL_class_t H5VL_pdc_g = {
    H5VL_VERSION,                       /* version */
    (H5VL_class_value_t)H5VL_PDC_VALUE, /* value */
    H5VL_PDC_NAME_STRING,               /* name */
    0,                                  /* connector version */
    H5VL_CAP_FLAG_ASYNC,                /* capability flags */
    H5VL_pdc_init,                      /* initialize */
    H5VL_pdc_obj_term,                  /* terminate */
    {
        /* info_cls */
        sizeof(H5VL_pdc_info_t), /* info size */
        H5VL_pdc_info_copy,      /* info copy */
        H5VL_pdc_info_cmp,       /* info compare */
        H5VL_pdc_info_free,      /* info free */
        H5VL_pdc_info_to_str,    /* to_str */
        H5VL_pdc_str_to_info,    /* from_str */
    },
    {
        /* wrap_cls */
        NULL, NULL, NULL, NULL, NULL,
        /* H5VL_pdc_get_object,    /1* get_object   *1/ */
        /* H5VL_pdc_get_wrap_ctx,  /1* get_wrap_ctx *1/ */
        /* H5VL_pdc_wrap_object,   /1* wrap_object  *1/ */
        /* H5VL_pdc_unwrap_object, /1* unwrap_object *1/ */
        /* H5VL_pdc_free_wrap_ctx  /1* free_wrap_ctx *1/ */
    },
    {
        /* attribute_cls */
        H5VL_pdc_attr_create,   /* create */
        H5VL_pdc_attr_open,     /* open */
        H5VL_pdc_attr_read,     /* read */
        H5VL_pdc_attr_write,    /* write */
        H5VL_pdc_attr_get,      /* get */
        H5VL_pdc_attr_specific, /* specific */
        H5VL_pdc_attr_optional, /* optional */
        H5VL_pdc_attr_close     /* close */
    },
    {
        /* dataset_cls  */
        H5VL_pdc_dataset_create,   /* create */
        H5VL_pdc_dataset_open,     /* open */
        H5VL_pdc_dataset_read,     /* read */
        H5VL_pdc_dataset_write,    /* write */
        H5VL_pdc_dataset_get,      /* get */
        H5VL_pdc_dataset_specific, /* specific */
        H5VL_pdc_dataset_optional, /* optional */
        H5VL_pdc_dataset_close     /* close */
    },
    {
        /* datatype_cls */
        H5VL_pdc_datatype_commit,   /* commit */
        H5VL_pdc_datatype_open,     /* open */
        H5VL_pdc_datatype_get,      /* get_size */
        H5VL_pdc_datatype_specific, /* specific */
        H5VL_pdc_datatype_optional, /* optional */
        H5VL_pdc_datatype_close     /* close */
    },
    {
        /* file_cls     */
        H5VL_pdc_file_create,   /* create */
        H5VL_pdc_file_open,     /* open */
        H5VL_pdc_file_get,      /* get */
        H5VL_pdc_file_specific, /* specific */
        H5VL_pdc_file_optional, /* optional */
        H5VL_pdc_file_close     /* close */
    },
    {
        /* group_cls    */
        H5VL_pdc_group_create,   /* create */
        H5VL_pdc_group_open,     /* open */
        H5VL_pdc_group_get,      /* get */
        H5VL_pdc_group_specific, /* specific */
        H5VL_pdc_group_optional, /* optional */
        H5VL_pdc_group_close     /* close */
    },
    {
        /* link_cls     */
        H5VL_pdc_link_create,   /* create */
        H5VL_pdc_link_copy,     /* copy */
        H5VL_pdc_link_move,     /* move */
        H5VL_pdc_link_get,      /* get */
        H5VL_pdc_link_specific, /* specific */
        H5VL_pdc_link_optional  /* optional */
    },
    {
        /* object_cls   */
        H5VL_pdc_object_open,     /* open */
        H5VL_pdc_object_copy,     /* copy */
        H5VL_pdc_object_get,      /* get */
        H5VL_pdc_object_specific, /* specific */
        H5VL_pdc_object_optional  /* optional */
    },
    {
        /* introspect_cls */
        H5VL_pdc_introspect_get_conn_cls,  /* get_conn_cls */
        H5VL_pdc_introspect_get_cap_flags, /* get_cap_flags */
        H5VL_pdc_introspect_opt_query      /* opt_query */
    },
    {
        /* request_cls  */
        H5VL_pdc_request_wait,     /* wait */
        H5VL_pdc_request_notify,   /* notify */
        H5VL_pdc_request_cancel,   /* cancel */
        H5VL_pdc_request_specific, /* specific */
        H5VL_pdc_request_optional, /* optional */
        H5VL_pdc_request_free      /* free */
    },
    {
        /* blob_cls */
        H5VL_pdc_blob_put,      /* put */
        H5VL_pdc_blob_get,      /* get */
        H5VL_pdc_blob_specific, /* specific */
        H5VL_pdc_blob_optional  /* optional */
    },
    {
        /* token_cls */
        H5VL_pdc_token_cmp,     /* cmp */
        H5VL_pdc_token_to_str,  /* to_str */
        H5VL_pdc_token_from_str /* from_str */
    },
    H5VL_pdc_optional /* optional */
};

/* The connector identification number, initialized at runtime */
static hid_t   H5VL_PDC_g      = H5I_INVALID_HID;
static hbool_t H5VL_pdc_init_g = FALSE;

/* Error stack declarations */
hid_t H5VL_ERR_STACK_g = H5I_INVALID_HID;
hid_t H5VL_ERR_CLS_g   = H5I_INVALID_HID;

static pdcid_t pdc_id_g  = 0;
static int     my_rank_g = 0;

/*---------------------------------------------------------------------------*/

/**
 * Public definitions.
 */

/*---------------------------------------------------------------------------*/
void
replace_multi_slash(char *str)
{
    char *dest = str;
    while (*str != '\0') {
        while (*str == '/' && *(str + 1) == '/')
            str++;
        *dest++ = *str++;
    }
    *dest = '\0';
}

/*---------------------------------------------------------------------------*/
H5PL_type_t
H5PLget_plugin_type(void)
{
    return H5PL_TYPE_VOL;
}

/*---------------------------------------------------------------------------*/
const void *
H5PLget_plugin_info(void)
{
    return &H5VL_pdc_g;
}

/*---------------------------------------------------------------------------*/
herr_t
H5VLpdc_term(void)
{
    FUNC_ENTER_VOL(herr_t, SUCCEED)

    /* Terminate the plugin */
    if (H5VL_pdc_obj_term() < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CLOSEERROR, FAIL, "can't terminate PDC VOL connector");

done:
    FUNC_LEAVE_VOL
}

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_init(hid_t H5VL_ATTR_UNUSED vipl_id)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    FUNC_ENTER_VOL(herr_t, SUCCEED)

    /* Check whether initialized */
    if (H5VL_pdc_init_g)
        HGOTO_ERROR(H5E_VOL, H5E_CANTINIT, FAIL, "attempting to initialize connector twice");

    /* Create error stack */
    if ((H5VL_ERR_STACK_g = H5Ecreate_stack()) < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CANTCREATE, FAIL, "can't create error stack");

    /* Register error class for error reporting */
    if ((H5VL_ERR_CLS_g =
             H5Eregister_class(H5VL_PDC_PACKAGE_NAME, H5VL_PDC_LIBRARY_NAME, H5VL_PDC_VERSION_STRING)) < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CANTREGISTER, FAIL, "can't register error class");

    /* Init PDC */
    if (pdc_id_g == 0) {
        pdc_id_g = PDCinit("pdc");
        if (pdc_id_g <= 0)
            HGOTO_ERROR(H5E_VOL, H5E_CANTINIT, FAIL, "could not initialize PDC");
        /* Initialized */
        H5VL_pdc_init_g = TRUE;
    }

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_init() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_obj_term(void)
{
    FUNC_ENTER_VOL(herr_t, SUCCEED)

    if (!H5VL_pdc_init_g)
        HGOTO_DONE(SUCCEED);

    if (pdc_id_g > 0 && PDCclose(pdc_id_g) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, FAIL, "failed to close PDC");
    pdc_id_g = 0;

    /* "Forget" plugin id.  This should normally be called by the library
     * when it is closing the id, so no need to close it here. */
    H5VL_PDC_g = H5I_INVALID_HID;

    H5VL_pdc_init_g = FALSE;

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_obj_term() */

/*---------------------------------------------------------------------------*/
static H5VL_pdc_obj_t *
H5VL_pdc_new_obj(void *under_obj, hid_t under_vol_id)
{
    H5VL_pdc_obj_t *new_obj;

    new_obj               = (H5VL_pdc_obj_t *)calloc(1, sizeof(H5VL_pdc_obj_t));
    new_obj->under_object = under_obj;
    new_obj->under_vol_id = under_vol_id;
    /* H5Iinc_ref(new_obj->under_vol_id); */

    return new_obj;
} /* end H5VL__pdc_new_obj() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_free_obj(H5VL_pdc_obj_t *obj)
{
    hid_t err_id;

    err_id = H5Eget_current_stack();
    H5Eset_current_stack(err_id);
    free(obj);

    return 0;
} /* end H5VL__pdc_free_obj() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_info_copy(const void *_old_info)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    const H5VL_pdc_info_t *old_info = (const H5VL_pdc_info_t *)_old_info;
    H5VL_pdc_info_t *      new_info = NULL;

    FUNC_ENTER_VOL(void *, NULL)

    if (NULL == (new_info = (H5VL_pdc_info_t *)malloc(sizeof(H5VL_pdc_info_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");

    new_info->under_vol_info = old_info->under_vol_info;
    new_info->under_vol_id   = old_info->under_vol_id;

    FUNC_RETURN_SET(new_info);

done:
    if (FUNC_ERRORED) {
        /* cleanup */
        if (new_info && H5VL_pdc_info_free(new_info) < 0)
            HDONE_ERROR(H5E_PLIST, H5E_CANTFREE, NULL, "can't free fapl");
    }

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_info_copy() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_info_cmp(int *cmp_value, const void *_info1, const void *_info2)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    const H5VL_pdc_info_t *info1 = (const H5VL_pdc_info_t *)_info1;
    const H5VL_pdc_info_t *info2 = (const H5VL_pdc_info_t *)_info2;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(info1);
    assert(info2);

    *cmp_value = memcmp(info1, info2, sizeof(H5VL_pdc_info_t));

    FUNC_LEAVE_VOL
}

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_info_free(void *_info)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_info_t *info = (H5VL_pdc_info_t *)_info;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(info);

    /* free the struct */
    free(info);

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_info_free() */

static herr_t
H5VL_pdc_info_to_str(const void *_info, char **str)
{
    const H5VL_pdc_info_t *info              = (const H5VL_pdc_info_t *)_info;
    H5VL_class_value_t     under_value       = (H5VL_class_value_t)-1;
    char *                 under_vol_string  = NULL;
    size_t                 under_vol_str_len = 0;

    /* Get value and string for underlying VOL connector */
    H5VLget_value(info->under_vol_id, &under_value);
    H5VLconnector_info_to_str(info->under_vol_info, info->under_vol_id, &under_vol_string);

    /* Determine length of underlying VOL info string */
    if (under_vol_string)
        under_vol_str_len = strlen(under_vol_string);

    /* Allocate space for our info */
    *str = (char *)H5allocate_memory(32 + under_vol_str_len, (hbool_t)0);
    assert(*str);

    /* Encode our info
     * Normally we'd use snprintf() here for a little extra safety, but that
     * call had problems on Windows until recently. So, to be as platform-independent
     * as we can, we're using sprintf() instead.
     */
    sprintf(*str, "under_vol=%u;under_info={%s}", (unsigned)under_value,
            (under_vol_string ? under_vol_string : ""));

    return 0;
} /* end H5VL_pdc_obj_to_str() */

hid_t
H5VL_pdc_register(void)
{
    /* Singleton register the pass-through VOL connector ID */
    if (H5VL_PDC_g < 0)
        H5VL_PDC_g = H5VLregister_connector(&H5VL_pdc_g, H5P_DEFAULT);

    return H5VL_PDC_g;
} /* end H5VL_pdc_register() */

static herr_t
H5VL_pdc_str_to_info(const char *str, void **_info)
{
    H5VL_pdc_info_t *info;
    unsigned         under_vol_value;
    const char *     under_vol_info_start, *under_vol_info_end;
    hid_t            under_vol_id;
    void *           under_vol_info = NULL;

    /* Retrieve the underlying VOL connector value and info */
    sscanf(str, "under_vol=%u;", &under_vol_value);
    under_vol_id         = H5VLregister_connector_by_value((H5VL_class_value_t)under_vol_value, H5P_DEFAULT);
    under_vol_info_start = strchr(str, '{');
    under_vol_info_end   = strrchr(str, '}');
    assert(under_vol_info_end > under_vol_info_start);
    if (under_vol_info_end != (under_vol_info_start + 1)) {
        char *under_vol_info_str;

        under_vol_info_str = (char *)malloc((size_t)(under_vol_info_end - under_vol_info_start));
        memcpy(under_vol_info_str, under_vol_info_start + 1,
               (size_t)((under_vol_info_end - under_vol_info_start) - 1));
        *(under_vol_info_str + (under_vol_info_end - under_vol_info_start)) = '\0';

        H5VLconnector_str_to_info(under_vol_info_str, under_vol_id, &under_vol_info);

        free(under_vol_info_str);
    } /* end else */

    /* Allocate new pass-through VOL connector info and set its fields */
    info                 = (H5VL_pdc_info_t *)calloc(1, sizeof(H5VL_pdc_info_t));
    info->under_vol_id   = under_vol_id;
    info->under_vol_info = under_vol_info;

    /* Set return value */
    *_info = info;

    return 0;
} /* end H5VL_pdc_str_to_info() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_get_object(const void *obj)
{
    const H5VL_pdc_obj_t *o = (const H5VL_pdc_obj_t *)obj;
    return H5VLget_object(o->under_object, o->under_vol_id);
} /* end H5VL_pdc_get_object() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_get_wrap_ctx(const void *obj, void **wrap_ctx)
{
    const H5VL_pdc_obj_t *o_pdc        = (const H5VL_pdc_obj_t *)obj;
    hid_t                 under_vol_id = 0;
    void *                under_object = NULL;
    H5VL_pdc_wrap_ctx_t * new_wrap_ctx;

    /* Allocate new VOL object wrapping context for the pdc connector */
    new_wrap_ctx = (H5VL_pdc_wrap_ctx_t *)calloc(1, sizeof(H5VL_pdc_wrap_ctx_t));
    if (new_wrap_ctx == NULL) {
        fprintf(stderr, "  [PDC VOL ERROR] with allocation in %s\n", __func__);
        return -1;
    }

    if (o_pdc->under_vol_id > 0) {
        under_vol_id = o_pdc->under_vol_id;
    }

    /* Increment reference count on underlying VOL ID, and copy the VOL info */
    new_wrap_ctx->under_vol_id = under_vol_id;
    H5Iinc_ref(new_wrap_ctx->under_vol_id);

    under_object = o_pdc->under_object;
    if (under_object) {
        H5VLget_wrap_ctx(under_object, under_vol_id, &new_wrap_ctx->under_wrap_ctx);
    }

    /* Set wrap context to return */
    *wrap_ctx = new_wrap_ctx;

    return 0;
} /* end H5VL_pdc_get_wrap_ctx() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_wrap_object(void *obj, H5I_type_t obj_type, void *_wrap_ctx)
{
    H5VL_pdc_wrap_ctx_t *wrap_ctx = (H5VL_pdc_wrap_ctx_t *)_wrap_ctx;
    H5VL_pdc_obj_t *     new_obj;
    void *               under;

    /* Wrap the object with the underlying VOL */
    under = H5VLwrap_object(obj, obj_type, wrap_ctx->under_vol_id, wrap_ctx->under_wrap_ctx);
    if (under) {
        if ((new_obj = H5VL_pdc_new_obj(under, wrap_ctx->under_vol_id)) == NULL) {
            fprintf(stderr, "  [PDC VOL ERROR] %s with request object calloc\n", __func__);
            return NULL;
        }
    }
    else
        new_obj = NULL;

    return new_obj;
} /* end H5VL_pdc_wrap_object() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_unwrap_object(void *obj)
{
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    void *          under;

    /* Unrap the object with the underlying VOL */
    under = H5VLunwrap_object(o->under_object, o->under_vol_id);

    if (under)
        H5VL_pdc_free_obj(o);

    return under;
} /* end H5VL_pdc_unwrap_object() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_free_wrap_ctx(void *_wrap_ctx)
{
    H5VL_pdc_wrap_ctx_t *wrap_ctx = (H5VL_pdc_wrap_ctx_t *)_wrap_ctx;
    hid_t                err_id;

    err_id = H5Eget_current_stack();

    /* Release underlying VOL ID and wrap context */
    if (wrap_ctx->under_wrap_ctx)
        H5VLfree_wrap_ctx(wrap_ctx->under_wrap_ctx, wrap_ctx->under_vol_id);
    H5Idec_ref(wrap_ctx->under_vol_id);

    H5Eset_current_stack(err_id);

    /* Free pdc wrap context object itself */
    free(wrap_ctx);

    return 0;
} /* end H5VL_pdc_free_wrap_ctx() */

/*---------------------------------------------------------------------------*/
static H5VL_pdc_obj_t *
H5VL__pdc_file_init(const char *name, unsigned flags __attribute__((unused)),
                    H5VL_pdc_info_t *info __attribute__((unused)), hid_t fapl_id)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *file = NULL;
    hid_t           under_vol_id, driver;

    FUNC_ENTER_VOL(void *, NULL)

    H5Pget_vol_id(fapl_id, &under_vol_id);

    /* allocate the file object that is returned to the user */
    if (NULL == (file = calloc(1, sizeof(H5VL_pdc_obj_t))))
        HGOTO_ERROR(H5E_FILE, H5E_CANTALLOC, NULL, "can't allocate PDC file struct");
    memset(file, 0, sizeof(H5VL_pdc_obj_t));
    file->info = MPI_INFO_NULL;
    file->comm = MPI_COMM_NULL;

    /* Fill in fields of file we know */
    file->under_object = file;
    file->under_vol_id = under_vol_id;
    file->h5i_type     = H5I_FILE;
    /* file->h5o_type     = H5O_TYPE_FILE; */
    file->file_obj_ptr = file;
    file->req_cnt      = 0;

    if (NULL == (file->file_name = strdup(name)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, NULL, "can't copy file name");

    driver = H5Pget_driver(fapl_id);
    if (driver == H5FD_MPIO) {
#ifdef ENABLE_LOGGING
        fprintf(stderr, "Rank %d: %s MPI-IO driver detected\n", my_rank_g, __func__);
#endif
        /* Duplicate communicator and Info object. */
        H5Pget_fapl_mpio(fapl_id, &file->comm, &file->info);

        /* Obtain the process rank and size from the communicator attached to the
         * fapl ID */
        //
        MPI_Comm_rank(file->comm, &file->my_rank);
        MPI_Comm_size(file->comm, &file->num_procs);
    }
    else {
#ifdef ENABLE_LOGGING
        fprintf(stderr, "Rank %d: %s non-MPI-IO driver\n", my_rank_g, __func__);
#endif
    }

    my_rank_g  = file->my_rank;
    file->nobj = 0;

    H5_LIST_INIT(&file->ids);

    FUNC_RETURN_SET((void *)file);

done:
    FUNC_LEAVE_VOL
} /* end H5VL__pdc_file_init() */

/*---------------------------------------------------------------------------*/
static herr_t _flush_xfer_requests(H5VL_pdc_obj_t *file);

static herr_t
H5VL__pdc_file_close(H5VL_pdc_obj_t *file)
{
    perr_t ret;
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(file);

    // Complete existing write requests
    ret = _flush_xfer_requests(file);
    if (ret != SUCCEED)
        HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "Failed to complete deferred writes");
    if (file->req_alloc) {
        free(file->xfer_requests);
        free(file->bufs);
        file->req_alloc = 0;
    }

    /* Free file data structures */
    if (file->file_name)
        free(file->file_name);
    if (file->comm != MPI_COMM_NULL)
        MPI_Comm_free(&file->comm);
    file->comm = MPI_COMM_NULL;

    free(file);
    file = NULL;

done:
    FUNC_LEAVE_VOL
} /* end H5VL__pdc_file_close() */

/*---------------------------------------------------------------------------*/
static H5VL_pdc_obj_t *
H5VL__pdc_dset_init(H5VL_pdc_obj_t *file)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *dset = NULL;

    FUNC_ENTER_VOL(void *, NULL)

    /* Allocate the dataset object that is returned to the user */
    if (NULL == (dset = calloc(1, sizeof(H5VL_pdc_obj_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, NULL, "can't allocate PDC dataset struct");
    memset(dset, 0, sizeof(H5VL_pdc_obj_t));

    dset->reg_id_from  = 0;
    dset->reg_id_to    = 0;
    dset->mapped       = 0;
    dset->type_id      = 0;
    dset->space_id     = 0;
    dset->h5i_type     = H5I_DATASET;
    dset->h5o_type     = H5O_TYPE_DATASET;
    dset->file_obj_ptr = file->file_obj_ptr;

    /* Set return value */
    FUNC_RETURN_SET((void *)dset);

done:
    FUNC_LEAVE_VOL
} /* end H5VL__pdc_dset_init() */

/* Integers map to the PDC type of the same width and signedness. The original
 * mapping sent every integer to PDC_INT (4 bytes), so 1-, 2- and 8-byte integer
 * datasets moved the wrong number of bytes. */
static pdc_var_type_t
_h5_int_to_pdc(hid_t type_id)
{
    int    is_signed = H5Tget_sign(type_id) != H5T_SGN_NONE;
    size_t size      = H5Tget_size(type_id);
    switch (size) {
        case 1:
            return is_signed ? PDC_INT8 : PDC_UINT8;
        case 2:
            return is_signed ? PDC_INT16 : PDC_UINT16;
        case 8:
            return is_signed ? PDC_INT64 : PDC_UINT64;
        default:
            return is_signed ? PDC_INT : PDC_UINT;
    }
}

static int
_pdc_is_int(pdc_var_type_t t)
{
    return t == PDC_INT || t == PDC_UINT || t == PDC_INT8 || t == PDC_UINT8 || t == PDC_INT16 ||
           t == PDC_UINT16 || t == PDC_INT32 || t == PDC_INT64 || t == PDC_UINT64;
}

/* HDF5 datatype for a PDC object type (H5Dget_type on reopened datasets) */
static hid_t
_pdc_to_h5_type(pdc_var_type_t t)
{
    switch (t) {
        case PDC_INT8: return H5Tcopy(H5T_NATIVE_INT8);
        case PDC_UINT8: return H5Tcopy(H5T_NATIVE_UINT8);
        case PDC_INT16: return H5Tcopy(H5T_NATIVE_INT16);
        case PDC_UINT16: return H5Tcopy(H5T_NATIVE_UINT16);
        case PDC_INT:
        case PDC_INT32: return H5Tcopy(H5T_NATIVE_INT32);
        case PDC_UINT: return H5Tcopy(H5T_NATIVE_UINT32);
        case PDC_INT64: return H5Tcopy(H5T_NATIVE_INT64);
        case PDC_UINT64: return H5Tcopy(H5T_NATIVE_UINT64);
        case PDC_FLOAT: return H5Tcopy(H5T_NATIVE_FLOAT);
        case PDC_DOUBLE: return H5Tcopy(H5T_NATIVE_DOUBLE);
        case PDC_CHAR:
        case PDC_STRING: return H5Tcopy(H5T_NATIVE_CHAR);
        default: return H5Tcopy(H5T_NATIVE_UINT8);
    }
}

/*---------------------------------------------------------------------------*/
static herr_t
H5VL__pdc_dset_free(H5VL_pdc_obj_t *dset)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    H5Pclose(dset->dcpl_id);
    H5Pclose(dset->dapl_id);
    H5Pclose(dset->dxpl_id);
    if (dset->type_id != 0)
        H5Tclose(dset->type_id);
    if (dset->space_id != 0 && dset->space_id != H5S_ALL)
        H5Sclose(dset->space_id);

    H5_LIST_REMOVE(dset, entry);
    free(dset);
    dset = NULL;

    FUNC_LEAVE_VOL
} /* end H5VL__pdc_dset_free() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL__pdc_sel_to_recx_iov(hid_t space_id, size_t type_size, uint64_t *off)
{
    hid_t   sel_iter_id;           /* Selection iteration info */
    hbool_t sel_iter_init = FALSE; /* Selection iteration info has been initialized */
    size_t  nseq;
    size_t  nelem;
    size_t  len[H5VL_PDC_SEQ_LIST_LEN];

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    /* Initialize selection iterator  */
    if ((sel_iter_id = H5Ssel_iter_create(space_id, type_size, 0)) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL, "unable to initialize selection iterator");
    sel_iter_init = TRUE; /* Selection iteration info has been initialized */

    /* Generate sequences from the file space until finished */
    do {
        /* Get the sequences of bytes */
        if (H5Ssel_iter_get_seq_list(sel_iter_id, (size_t)H5VL_PDC_SEQ_LIST_LEN, (size_t)-1, &nseq, &nelem,
                                     off, len) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "sequence length generation failed");
    } while (nseq == H5VL_PDC_SEQ_LIST_LEN);

done:
    /* Release selection iterator */
    if (sel_iter_init && H5Ssel_iter_close(sel_iter_id) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to release selection iterator");
    FUNC_LEAVE_VOL
} /* end H5VL__pdc_sel_to_recx_iov() */

/*---------------------------------------------------------------------------*/
void *
H5VL_pdc_file_create(const char *name, unsigned flags, hid_t fcpl_id __attribute__((unused)), hid_t fapl_id,
                     hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s, fname [%s]\n", my_rank_g, __func__, name);
#endif

    H5VL_pdc_info_t *info;
    H5VL_pdc_obj_t * file = NULL;
    pdcid_t          cont_prop;

    FUNC_ENTER_VOL(void *, NULL)

    assert(name);

    /* Get information from the FAPL */
    if (H5Pget_vol_info(fapl_id, (void **)&info) < 0)
        HGOTO_ERROR(H5E_SYM, H5E_CANTGET, NULL, "can't get PDC info struct");

    /* Initialize file */
    if (NULL == (file = H5VL__pdc_file_init(name, flags, info, fapl_id)))
        HGOTO_ERROR(H5E_FILE, H5E_CANTINIT, NULL, "can't init PDC file struct");

    if ((cont_prop = PDCprop_create(PDC_CONT_CREATE, pdc_id_g)) <= 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCREATE, NULL, "can't create container property");

    if ((file->cont_id = PDCcont_create(name, cont_prop)) <= 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCREATE, NULL, "can't create container");

    if ((PDCprop_close(cont_prop)) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCREATE, NULL, "can't close container property");

    /* Free info */
    if (info && H5VL_pdc_info_free(info) < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CANTFREE, NULL, "can't free connector info");

    FUNC_RETURN_SET((void *)file);

done:
    if (FUNC_ERRORED) {
        /* Free info */
        if (info)
            if (H5VL_pdc_info_free(info) < 0)
                HDONE_ERROR(H5E_VOL, H5E_CANTFREE, NULL, "can't free connector info");

        /* Close file */
        if (file == NULL)
            HDONE_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, NULL, "can't close file");
    } /* end if */

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_file_create() */

/*---------------------------------------------------------------------------*/
void *
H5VL_pdc_file_open(const char *name, unsigned flags, hid_t fapl_id, hid_t dxpl_id __attribute__((unused)),
                   void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_info_t *info;
    H5VL_pdc_obj_t * file = NULL;

    FUNC_ENTER_VOL(void *, NULL)

    /* Get information from the FAPL */
    if (H5Pget_vol_info(fapl_id, (void **)&info) < 0)
        HGOTO_ERROR(H5E_SYM, H5E_CANTGET, NULL, "can't get PDC info struct");

    /* Initialize file */
    if (NULL == (file = H5VL__pdc_file_init(name, flags, info, fapl_id)))
        HGOTO_ERROR(H5E_FILE, H5E_CANTINIT, NULL, "can't init PDC file struct");

    if ((file->cont_id = PDCcont_open(name, pdc_id_g)) <= 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTOPENFILE, NULL, "failed to create container");

    /* Free info */
    if (info && H5VL_pdc_info_free(info) < 0)
        HGOTO_ERROR(H5E_VOL, H5E_CANTFREE, NULL, "can't free connector info");

    FUNC_RETURN_SET((void *)file);

done:
    if (FUNC_ERRORED) {
        /* Free info */
        if (info)
            if (H5VL_pdc_info_free(info) < 0)
                HGOTO_ERROR(H5E_VOL, H5E_CANTFREE, NULL, "can't free connector info");
        if (file == NULL)
            HDONE_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, NULL, "can't close file");
    } /* end if */

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_file_open() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_file_close(void *_file, hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *file = (H5VL_pdc_obj_t *)_file;
    /* H5VL_pdc_obj_t *dset = NULL; */
    perr_t ret;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(file);

    /* while (!H5_LIST_IS_EMPTY(&file->ids)) { */
    /*     H5_LIST_GET_FIRST(dset, &file->ids); */
    /*     H5_LIST_REMOVE(dset, entry); */
    /*     if (H5VL__pdc_dset_free(dset) < 0) */
    /*         HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "failed to free dataset"); */
    /* } */

    if ((ret = PDCcont_close(file->cont_id)) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, FAIL, "failed to close container");

    /* Close the file */
    if (H5VL__pdc_file_close(file) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, FAIL, "failed to close file");

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_file_close() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_file_specific(void *file, H5VL_file_specific_args_t *args, hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *           o = (H5VL_pdc_obj_t *)file;
    H5VL_pdc_obj_t *           new_o;
    H5VL_file_specific_args_t  my_args;
    H5VL_file_specific_args_t *new_args;
    H5VL_pdc_info_t *          info;
    hid_t                      under_vol_id = -1;
    herr_t                     ret_value    = 0;

    if (args->op_type == H5VL_FILE_IS_ACCESSIBLE) {

        /* Shallow copy the args */

        memcpy(&my_args, args, sizeof(my_args));

        /* Get copy of our VOL info from FAPL */
        H5Pget_vol_info(args->args.is_accessible.fapl_id, (void **)&info);
        /* Make sure we have info about the underlying VOL to be used */
        if (!info)
            return (-1);

        /* Keep the correct underlying VOL ID for later */
        under_vol_id = info->under_vol_id;

        /* Copy the FAPL */
        my_args.args.is_accessible.fapl_id = H5Pcopy(args->args.is_accessible.fapl_id);

        /* Set the VOL ID and info for the underlying FAPL */
        H5Pset_vol(my_args.args.is_accessible.fapl_id, info->under_vol_id, info->under_vol_info);

        /* Set argument pointer to new arguments */
        new_args = &my_args;

        /* Set object pointer for operation */
        new_o = NULL;
    } /* end else-if */
    else if (args->op_type == H5VL_FILE_DELETE) {
        /* Shallow copy the args */
        memcpy(&my_args, args, sizeof(my_args));

        /* Get copy of our VOL info from FAPL */
        H5Pget_vol_info(args->args.del.fapl_id, (void **)&info);
        /* Make sure we have info about the underlying VOL to be used */
        if (!info)
            return (-1);

        /* Keep the correct underlying VOL ID for later */
        under_vol_id = info->under_vol_id;

        /* Copy the FAPL */
        my_args.args.del.fapl_id = H5Pcopy(args->args.del.fapl_id);

        /* Set the VOL ID and info for the underlying FAPL */
        H5Pset_vol(my_args.args.del.fapl_id, info->under_vol_id, info->under_vol_info);

        /* Set argument pointer to new arguments */
        new_args = &my_args;

        /* Set object pointer for operation */
        new_o = NULL;
    } /* end else-if */
    else {
        /* Keep the correct underlying VOL ID for later */
        under_vol_id = o->under_vol_id;

        /* Set argument pointer to current arguments */
        new_args = args;

        /* Set object pointer for operation */
        new_o = o->under_object;
    } /* end else */
    if (args->op_type == H5VL_FILE_FLUSH) {
        /* the deferred write queue lives on the file object */
        H5VL_pdc_obj_t *f = o ? o->file_obj_ptr : NULL;
        ret_value = _flush_xfer_requests(f ? f : o) == SUCCEED ? 0 : -1;
    }
    else {
        ret_value = H5VLfile_specific(new_o, under_vol_id, new_args, dxpl_id, req);
    }

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, under_vol_id);

    if (args->op_type == H5VL_FILE_IS_ACCESSIBLE) {
        /* Close underlying FAPL */
        H5Pclose(my_args.args.is_accessible.fapl_id);

        /* Release copy of our VOL info */
        H5VL_pdc_info_free(info);
    } /* end else-if */
    else if (args->op_type == H5VL_FILE_DELETE) {
        /* Close underlying FAPL */
        H5Pclose(my_args.args.del.fapl_id);

        /* Release copy of our VOL info */
        H5VL_pdc_info_free(info);
    } /* end else-if */
    else if (args->op_type == H5VL_FILE_REOPEN) {
        /* Wrap file struct pointer for 'reopen' operation, if we reopened one */
        if (ret_value >= 0 && *args->args.reopen.file)
            *args->args.reopen.file = H5VL_pdc_new_obj(*args->args.reopen.file, under_vol_id);
    } /* end else */

    return ret_value;
} /* end H5VL_pdc_file_specific() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_file_get(void *file __attribute__((unused)), H5VL_file_get_args_t *args __attribute__((unused)),
                  hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    return 0;
} /* end H5VL_pdc_file_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_file_optional(void *file __attribute__((unused)), H5VL_optional_args_t *args __attribute__((unused)),
                       hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    return 0;
} /* end H5VL_pdc_file_optional() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_dataset_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                        hid_t lcpl_id __attribute__((unused)), hid_t type_id, hid_t space_id, hid_t dcpl_id,
                        hid_t dapl_id, hid_t dxpl_id, void **req __attribute__((unused)))
{
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    int             buff_len, ndim;
    H5T_class_t     dclass;
    H5VL_pdc_obj_t *dset = NULL;
    pdcid_t         obj_prop, obj_id;
    hsize_t         dims[H5S_MAX_RANK];

    FUNC_ENTER_VOL(void *, NULL)

#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entered dataset_create [%s][%s][%s]\n", o->my_rank, o->file_name, o->group_name,
            name);
#endif

    if (o->group_name) {
        buff_len = strlen(name) + strlen(o->file_name) + strlen(o->group_name) + 3;
    }
    else {
        buff_len = strlen(name) + strlen(o->file_name) + 2;
    }
    char new_name[buff_len];
    if (o->group_name) {
        strcpy(new_name, name);
        strcat(new_name, "/");
        strcat(new_name, o->group_name);
        strcat(new_name, "/");
        strcat(new_name, o->file_name);
    }
    else {
        strcpy(new_name, name);
        strcat(new_name, "/");
        strcat(new_name, o->file_name);
    }
    /* Assume that the name, group name, and file_name do not include multiple consecutive
       slashes as a part of their names. */
    replace_multi_slash(new_name);

    if (!obj)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "parent object is NULL");
    if (!loc_params)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "location parameters object is NULL");
    if (!name)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "dataset name is NULL");

    /* Init dataset */
    if (NULL == (dset = H5VL__pdc_dset_init(o)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, NULL, "can't init PDC dataset struct");

    /* Finish setting up dataset struct */
    if ((dset->type_id = H5Tcopy(type_id)) < 0)
        HGOTO_ERROR(H5E_SYM, H5E_CANTCOPY, NULL, "failed to copy datatype");
    if ((dset->space_id = H5Scopy(space_id)) < 0)
        HGOTO_ERROR(H5E_SYM, H5E_CANTCOPY, NULL, "failed to copy dataspace");
    if (H5Sselect_all(dset->space_id) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTDELETE, NULL, "can't change selection");
    dset->dcpl_id = H5Pcopy(dcpl_id);
    dset->dapl_id = H5Pcopy(dapl_id);
    dset->dxpl_id = H5Pcopy(dxpl_id);

    obj_prop = PDCprop_create(PDC_OBJ_CREATE, pdc_id_g);

    dclass = H5Tget_class(type_id);
    switch (dclass) {
        case H5T_INTEGER:
            dset->pdc_type = _h5_int_to_pdc(type_id);
            PDCprop_set_obj_type(obj_prop, dset->pdc_type);
            break;
        case H5T_FLOAT:
            /* printf("Datatype class: Float\n"); */
            if (H5Tequal(H5T_NATIVE_DOUBLE, type_id) == TRUE) {
                PDCprop_set_obj_type(obj_prop, PDC_DOUBLE);
                dset->pdc_type = PDC_DOUBLE;
            }
            else {
                PDCprop_set_obj_type(obj_prop, PDC_FLOAT);
                dset->pdc_type = PDC_FLOAT;
            }
            break;
        case H5T_STRING:
            /* printf("Datatype class: String\n"); */
            PDCprop_set_obj_type(obj_prop, PDC_STRING);
            dset->pdc_type = PDC_STRING;
            break;
        case H5T_COMPOUND:
            PDCprop_set_obj_type(obj_prop, PDC_CHAR);
            dset->pdc_type = PDC_CHAR;
            break;
        case H5T_ARRAY:
            printf("Datatype class: Array is not supported in PDC\n");
            break;
        case H5T_ENUM:
            dset->pdc_type = _h5_int_to_pdc(type_id);
            PDCprop_set_obj_type(obj_prop, dset->pdc_type);
            break;
        case H5T_REFERENCE:
            printf("Datatype class: Reference is not supported in PDC\n");
            break;
        case H5T_OPAQUE:
            printf("Datatype class: Opaque is not supported in PDC\n");
            break;
        case H5T_NO_CLASS:
        default:
            printf("Unknown or no datatype class\n");
            break;
    }

    /* Get dataspace extent */
    if ((ndim = H5Sget_simple_extent_ndims(space_id)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, NULL, "can't get number of dimensions");
    if (ndim != H5Sget_simple_extent_dims(space_id, dims, NULL))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, NULL, "can't get dimensions");

    // TODO: temporary workaround for writing compound data, as current PDC doesn't support
    //       compound datatype.
    //       Multiple the last dimension by the compound dtype size so we can write the
    //       correct amount of total data, and add a tag to record for future read.
    if (dclass == H5T_COMPOUND) {
        o->compound_size = H5Tget_size(type_id);
        dims[ndim - 1] *= o->compound_size;
    }

    PDCprop_set_obj_dims(obj_prop, ndim, dims);

    /* Create PDC object */
    if (o->comm != MPI_COMM_NULL) {
#ifdef ENABLE_LOGGING
        fprintf(stderr, "Rank %d: PDC obj create mpi [%s]\n", o->my_rank, new_name);
#endif
        obj_id = PDCobj_create_mpi(o->cont_id, new_name, obj_prop, 0, o->comm);
    }
    else {
#ifdef ENABLE_LOGGING
        fprintf(stderr, "Rank %d: PDC obj create [%s]\n", o->my_rank, new_name);
#endif
        obj_id = PDCobj_create(o->cont_id, new_name, obj_prop);
    }

#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: PDC obj id %lu, dims %lu\n", o->my_rank, obj_id, dims[0]);
#endif

    // TODO: temporary workaround for writing compound data, as current PDC doesn't support
    //       compound datatype
    if (dclass == H5T_COMPOUND)
        PDCobj_put_tag(obj_id, "PDC_COMPOUND_DTYPE_SIZE", (void *)&o->compound_size, PDC_SIZE_T,
                       sizeof(psize_t));

    dset->obj_id   = obj_id;
    dset->h5i_type = H5I_DATASET;
    dset->h5o_type = H5O_TYPE_DATASET;
    strcpy(dset->obj_name, name);
    o->nobj++;
    H5_LIST_INSERT_HEAD(&o->ids, dset, entry);

    if ((PDCprop_close(obj_prop)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, NULL, "can't close object property");

    /* Set return value */
    FUNC_RETURN_SET((void *)dset);

done:
    FUNC_LEAVE_VOL
}

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_dataset_open(void *obj, const H5VL_loc_params_t *loc_params, const char *_name,
                      hid_t dapl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                      void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    FUNC_ENTER_VOL(void *, NULL)

    H5VL_pdc_obj_t *     o    = (H5VL_pdc_obj_t *)obj;
    H5VL_pdc_obj_t *     dset = NULL;
    struct pdc_obj_info *obj_info;

    int buff_len;
    if (o->group_name) {
        buff_len = strlen(_name) + strlen(o->file_name) + strlen(o->group_name) + 3;
    }
    else {
        buff_len = strlen(_name) + strlen(o->file_name) + 2;
    }

    char name[buff_len];
    if (o->group_name) {
        strcpy(name, _name);
        strcat(name, "/");
        strcat(name, o->group_name);
        strcat(name, "/");
        strcat(name, o->file_name);
    }
    else {
        strcpy(name, _name);
        strcat(name, "/");
        strcat(name, o->file_name);
    }
    /* Assume that the name, group name, and file_name do not include multiple consecutive
       slashes as a part of their names. */
    replace_multi_slash(name);

    if (!obj)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "parent object is NULL");
    if (!loc_params)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "location parameters object is NULL");
    if (0 == strlen(name))
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "dataset name is NULL");

    /* Init dataset */
    if (NULL == (dset = H5VL__pdc_dset_init(o)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, NULL, "can't init PDC dataset struct");

    strcpy(dset->obj_name, name);
    dset->obj_id = PDCobj_open(name, pdc_id_g);
    if (dset->obj_id <= 0) {
        free(dset);
        return NULL;
    }
    dset->under_vol_id = o->under_vol_id;
    dset->under_object = dset;
    /* pdcid_t id_name    = (pdcid_t)name; */
    obj_info       = PDCobj_get_info(dset->obj_id);
    dset->pdc_type = obj_info->obj_pt->type;

    // TODO: temporary workaround for writing compound data, as current PDC doesn't support
    //       compound datatype
    if (dset->pdc_type == PDC_CHAR) {
        psize_t        value_size;
        pdc_var_type_t value_type;
        psize_t *      value;
        PDCobj_get_tag(dset->obj_id, "PDC_COMPOUND_DTYPE_SIZE", (void **)&value, &value_type, &value_size);
        if (value_size > 0) {
            dset->compound_size = *value;
            obj_info->obj_pt->dims[obj_info->obj_pt->ndim - 1] /= *value;
        }
    }

    dset->space_id = H5Screate_simple(obj_info->obj_pt->ndim, obj_info->obj_pt->dims, NULL);
    if (dset->type_id <= 0)
        dset->type_id = _pdc_to_h5_type(dset->pdc_type);
    o->nobj++;
    H5_LIST_INSERT_HEAD(&o->ids, dset, entry);

    /* Set return value */
    FUNC_RETURN_SET((void *)dset);

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_open() */

int
_check_mem_type_id(H5T_class_t h5_dclass, pdc_var_type_t pdc_dtype)
{
    int is_match = 0;
    switch (h5_dclass) {
        case H5T_INTEGER:
            if (_pdc_is_int(pdc_dtype))
                is_match = 1;
            break;
        case H5T_FLOAT:
            if (pdc_dtype == PDC_FLOAT || pdc_dtype == PDC_DOUBLE)
                is_match = 1;
            break;
        case H5T_STRING:
            if (pdc_dtype == PDC_CHAR || pdc_dtype == PDC_STRING)
                is_match = 1;
            break;
        case H5T_COMPOUND:
            // TODO: temp workaround
            if (pdc_dtype == PDC_CHAR)
                is_match = 1;
            break;
        case H5T_ARRAY:
            printf("Datatype class: Array is not supported in PDC\n");
            break;
        case H5T_ENUM:
            if (_pdc_is_int(pdc_dtype))
                is_match = 1;
            break;
        case H5T_REFERENCE:
            printf("Datatype class: Reference is not supported in PDC\n");
            break;
        case H5T_OPAQUE:
            printf("Datatype class: Opaque is not supported in PDC\n");
            break;
        case H5T_NO_CLASS:
        default:
            printf("Unknown or no datatype class\n");
            break;
    }

    return is_match;
}

/*---------------------------------------------------------------------------*/
herr_t
_add_xfer_request(H5VL_pdc_obj_t *file, pdcid_t transfer_request, void *buf)
{
    if (file == NULL) {
        fprintf(stderr, "Rank %d: error with %s, empty file ptr\n", my_rank_g, __func__);
        return FAIL;
    }

    if (file->req_alloc == 0) {
        file->req_alloc     = 64;
        file->req_cnt       = 0;
        file->xfer_requests = (pdcid_t *)calloc(file->req_alloc, sizeof(pdcid_t));
        file->bufs          = (void **)calloc(file->req_alloc, sizeof(void *));
    }

    if (file->req_cnt > file->req_alloc - 2) {
        file->req_alloc *= 2;
        file->xfer_requests = (pdcid_t *)realloc(file->xfer_requests, file->req_alloc * sizeof(pdcid_t));
        file->bufs          = (void **)realloc(file->bufs, file->req_alloc * sizeof(void *));
    }

    file->xfer_requests[file->req_cnt] = transfer_request;
    file->bufs[file->req_cnt]          = buf;
    file->req_cnt++;

    return SUCCEED;
}

/* Start, wait for and close every deferred write of this file and free the
 * cached user buffers. Must run while the target PDC objects are still open. */
static herr_t
_flush_xfer_requests(H5VL_pdc_obj_t *file)
{
    herr_t ret_value = SUCCEED;

    if (file == NULL || file->req_cnt <= 0)
        return SUCCEED;
    if (PDCregion_transfer_start_all(file->xfer_requests, file->req_cnt) != SUCCEED)
        ret_value = FAIL;
    else if (PDCregion_transfer_wait_all(file->xfer_requests, file->req_cnt) != SUCCEED)
        ret_value = FAIL;
    for (int i = 0; i < file->req_cnt; i++) {
        if (PDCregion_transfer_close(file->xfer_requests[i]) != SUCCEED)
            ret_value = FAIL;
        file->xfer_requests[i] = 0;
        if (file->bufs[i] != NULL) {
            free(file->bufs[i]);
            file->bufs[i] = NULL;
        }
    }
    file->req_cnt      = 0;
    write_cache_size_g = 0;
    return ret_value;
}

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_dataset_write(size_t count, void *_dset[], hid_t mem_type_id[], hid_t mem_space_id[],
                       hid_t file_space_id[], hid_t plist_id __attribute__((unused)), const void *buf[],
                       void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *dset, *file;
    uint64_t        offset[H5S_MAX_RANK] = {0}, total_size = 0;
    size_t          type_size;
    int             ndim;
    pdcid_t         region_local, region_remote;
    hsize_t         dims[H5S_MAX_RANK] = {0};
    perr_t          ret;
    pdcid_t         transfer_request;
    H5T_class_t     h5_dclass;
    void *          cache_buf = NULL;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    for (size_t u = 0; u < count; u++) {
        dset = (H5VL_pdc_obj_t *)_dset[u];
        file = dset->file_obj_ptr;

#ifdef ENABLE_LOGGING
        fprintf(stderr, "Rank %d: writing [%s][%s][%s]\n", my_rank_g, dset->file_name, dset->group_name,
                dset->obj_name);
#endif
        if (file_space_id[u] == H5S_ALL)
            file_space_id[u] = dset->space_id;
        if (mem_space_id[u] == H5S_ALL)
            mem_space_id[u] = file_space_id[u];

        h5_dclass = H5Tget_class(mem_type_id[u]);
        if (_check_mem_type_id(h5_dclass, dset->pdc_type) == 0)
            HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "vol-pdc does not support datatype conversion");

        /* Get memory dataspace object */
        if ((ndim = H5Sget_simple_extent_ndims(mem_space_id[u])) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get number of dimensions");
        if (ndim != H5Sget_simple_extent_dims(mem_space_id[u], dims, NULL))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get dimensions");

        if (ndim > 4)
            HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "data dimension not supported");

        total_size = 1;
        for (int i = 0; i < ndim; i++)
            total_size *= dims[i];

        // TODO: temporary workaround for writing compound data, as current PDC doesn't support
        //       compound datatype
        type_size = H5Tget_size(mem_type_id[u]);
        if (H5Tget_class(mem_type_id[u]) == H5T_COMPOUND)
            dims[ndim - 1] *= type_size;

        total_size *= type_size;

        /* printf("Rank %d: mem offset %lu\n", dset->my_rank, offset[0]); */
        /* printf("Rank %d: mem count  %lu\n", dset->my_rank, dims[0]); */
        region_local      = PDCregion_create(ndim, offset, dims);
        dset->reg_id_from = region_local;

        /* H5VL__pdc_sel_to_recx_iov(file_space_id[u], type_size, offset); */
        H5VL__pdc_sel_to_recx_iov(file_space_id[u], 1, offset);

#ifdef ENABLE_LOGGING
        printf("Rank %d: file offset0 %lu, count0 %lu\n", my_rank_g, offset[0], dims[0]);
        if (ndim > 1)
            printf("Rank %d: file offset1 %lu, count1 %lu\n", my_rank_g, offset[1], dims[1]);
#endif
        region_remote   = PDCregion_create(ndim, offset, dims);
        dset->reg_id_to = region_remote;

        if (file->req_alloc == 0) {
            file->req_alloc     = 64;
            file->req_cnt       = 0;
            file->xfer_requests = (pdcid_t *)calloc(file->req_alloc, sizeof(pdcid_t));
            file->bufs          = (void **)calloc(file->req_alloc, sizeof(void *));
        }

        if (write_cache_size_g + total_size > MAX_WRITE_CACHE_SIZE_GB * 1073741824llu) {
            // Reaching max cache size, finish existing transfer requests and the current one
            transfer_request = PDCregion_transfer_create((void *)buf[u], PDC_WRITE, dset->obj_id,
                                                         region_local, region_remote);

            _add_xfer_request(file, transfer_request, NULL);

            ret = _flush_xfer_requests(file);
            if (ret != SUCCEED)
                HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "Failed to complete deferred writes");
        }
        else {
            // Cache the user buffer
            cache_buf = malloc(total_size);
            memcpy(cache_buf, buf[u], total_size);
            write_cache_size_g += total_size;

            transfer_request =
                PDCregion_transfer_create(cache_buf, PDC_WRITE, dset->obj_id, region_local, region_remote);

            _add_xfer_request(file, transfer_request, cache_buf);
        }

        // Defer xfer wait to the next read operation and file close time
    }

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_write() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_dataset_read(size_t count, void *_dset[], hid_t mem_type_id[], hid_t mem_space_id[],
                      hid_t file_space_id[], hid_t plist_id __attribute__((unused)), void *buf[],
                      void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *dset, *file;
    uint64_t        offset[H5S_MAX_RANK] = {0};
    int             ndim, i;
    pdcid_t         region_local, region_remote;
    hsize_t         dims[H5S_MAX_RANK] = {0};
    perr_t          ret;
    pdcid_t         transfer_request;
    H5T_class_t     h5_dclass;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    // Complete existing write requests
    dset = (H5VL_pdc_obj_t *)_dset[0];
    file = dset->file_obj_ptr;
    if (_flush_xfer_requests(file) != SUCCEED)
        HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "Failed to complete deferred writes");

    for (size_t u = 0; u < count; u++) {
        dset = (H5VL_pdc_obj_t *)_dset[u];

        h5_dclass = H5Tget_class(mem_type_id[u]);
        if (_check_mem_type_id(h5_dclass, dset->pdc_type) == 0)
            HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "vol-pdc does not support datatype conversion");

        /* Get memory dataspace object */
        if (mem_space_id[u] == H5S_ALL) {
            if ((ndim = H5Sget_simple_extent_ndims(dset->space_id)) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get number of dimensions");
            if (ndim != H5Sget_simple_extent_dims(dset->space_id, dims, NULL))
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get dimensions");
        }
        else {
            if ((ndim = H5Sget_simple_extent_ndims(mem_space_id[u])) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get number of dimensions");
            if (ndim != H5Sget_simple_extent_dims(mem_space_id[u], dims, NULL))
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get dimensions");
        }

        // TODO: temporary workaround for reading compound data, as current PDC doesn't support
        //       compound datatype
        if (dset->compound_size > 0) {
            dims[ndim - 1] *= dset->compound_size;
        }

        region_local      = PDCregion_create(ndim, offset, dims);
        dset->reg_id_from = region_local;

        if (file_space_id[u] != H5S_ALL)
            H5VL__pdc_sel_to_recx_iov(file_space_id[u], 1, offset);

        region_remote   = PDCregion_create(ndim, offset, dims);
        dset->reg_id_to = region_remote;

        transfer_request =
            PDCregion_transfer_create((void *)buf[u], PDC_READ, dset->obj_id, region_local, region_remote);
        ret = PDCregion_transfer_start(transfer_request);
        if (ret != SUCCEED) {
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "Failed to region transfer start");
        }
        ret = PDCregion_transfer_wait(transfer_request);
        if (ret != SUCCEED) {
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "Failed to region transfer wait");
        }
        ret = PDCregion_transfer_close(transfer_request);
        if (ret != SUCCEED) {
            HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "Failed to region transfer close");
        }
    } // End for u < count

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_read() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_dataset_get(void *_dset, H5VL_dataset_get_args_t *args, hid_t dxpl_id __attribute__((unused)),
                     void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *dset = (H5VL_pdc_obj_t *)_dset;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    H5VL_dataset_get_t get_type = (*args).op_type;

    switch (get_type) {
        case H5VL_DATASET_GET_DCPL: {
            args->args.get_dcpl.dcpl_id = H5Pcopy(dset->dcpl_id);
            break;
        }
        case H5VL_DATASET_GET_DAPL: {
            args->args.get_dapl.dapl_id = H5Pcopy(dset->dapl_id);
            break;
        }
        case H5VL_DATASET_GET_SPACE: {
            args->args.get_space.space_id = H5Scopy(dset->space_id);
            break;
        }
        case H5VL_DATASET_GET_TYPE: {
            args->args.get_type.type_id = H5Tcopy(dset->type_id);
            break;
        }
        case H5VL_DATASET_GET_STORAGE_SIZE:
        default:
            HGOTO_ERROR(H5E_VOL, H5E_UNSUPPORTED, FAIL, "can't get this type of information from dataset");
    } /* end switch */

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_dataset_specific(void *                        obj __attribute__((unused)),
                          H5VL_dataset_specific_args_t *args __attribute__((unused)),
                          hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_dataset_specific() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_dataset_optional(void *                obj __attribute__((unused)),
                          H5VL_optional_args_t *args __attribute__((unused)),
                          hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_dataset_optional() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_dataset_close(void *_dset, hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *dset = (H5VL_pdc_obj_t *)_dset;
    perr_t          ret;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(dset);
    if (_flush_xfer_requests(dset->file_obj_ptr) != SUCCEED)
        HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "Failed to complete deferred writes");
    if ((ret = PDCobj_close(dset->obj_id)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't close object");
    if (dset->reg_id_from != 0) {
        if ((ret = PDCregion_close(dset->reg_id_from)) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't close region");
    }
    if (dset->reg_id_to != 0) {
        if ((ret = PDCregion_close(dset->reg_id_to) < 0))
            HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't close remote region");
    }

    if (dset->mapped == 1)
        H5_LIST_REMOVE(dset, entry);

    if (H5VL__pdc_dset_free(dset) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't free dataset");

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_dataset_close() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_datatype_commit(void *                   obj __attribute__((unused)),
                         const H5VL_loc_params_t *loc_params __attribute__((unused)),
                         const char *name __attribute__((unused)), hid_t type_id __attribute__((unused)),
                         hid_t lcpl_id __attribute__((unused)), hid_t tcpl_id __attribute__((unused)),
                         hid_t tapl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                         void **req __attribute__((unused)))
{
    return (void *)0;
} /* end H5VL_pdc_datatype_commit() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_datatype_open(void *                   obj __attribute__((unused)),
                       const H5VL_loc_params_t *loc_params __attribute__((unused)),
                       const char *name __attribute__((unused)), hid_t tapl_id __attribute__((unused)),
                       hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return (void *)0;
} /* end H5VL_pdc_datatype_open() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_datatype_get(void *                    dt __attribute__((unused)),
                      H5VL_datatype_get_args_t *args __attribute__((unused)),
                      hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_datatype_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_datatype_specific(void *                         obj __attribute__((unused)),
                           H5VL_datatype_specific_args_t *args __attribute__((unused)),
                           hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_datatype_specific() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_datatype_optional(void *                obj __attribute__((unused)),
                           H5VL_optional_args_t *args __attribute__((unused)),
                           hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_datatype_optional() */

static herr_t
H5VL_pdc_datatype_close(void *dt __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                        void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_datatype_close() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl, const H5VL_class_t **conn_cls)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    herr_t          ret_value;

    /* Check for querying this connector's class */
    if (H5VL_GET_CONN_LVL_CURR == lvl) {
        *conn_cls = &H5VL_pdc_g;
        ret_value = 0;
    } /* end if */
    else
        ret_value = H5VLintrospect_get_conn_cls(o->under_object, o->under_vol_id, lvl, conn_cls);

    return ret_value;
} /* end H5VL_pdc_introspect_get_conn_cls() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_introspect_get_cap_flags(const void *_info, uint64_t *cap_flags)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    const H5VL_pdc_info_t *info = (const H5VL_pdc_info_t *)_info;
    herr_t                 ret_value;

    /* Invoke the query on the underlying VOL connector */
    ret_value = H5VLintrospect_get_cap_flags(info->under_vol_info, info->under_vol_id, cap_flags);

    /* Bitwise OR our capability flags in */
    if (ret_value >= 0)
        *cap_flags |= H5VL_pdc_g.cap_flags;

    return ret_value;
} /* end H5VL_pdc_introspect_get_cap_flags() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_introspect_opt_query(void *obj __attribute__((unused)), H5VL_subclass_t cls __attribute__((unused)),
                              int opt_type __attribute__((unused)), uint64_t *flags __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    /* H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj; */
    /* herr_t          ret_value; */
    return 0;
} /* end H5VL_pdc_introspect_opt_query() */

/*---------------------------------------------------------------------------*/
static void *
// store group name in pdc_obj
// retrieve file name and concatenate with group name, create pdc container (check code in file create)
H5VL_pdc_group_create(void *obj, const H5VL_loc_params_t *loc_params __attribute__((unused)),
                      const char *name, hid_t lcpl_id __attribute__((unused)),
                      hid_t gcpl_id __attribute__((unused)), hid_t gapl_id __attribute__((unused)),
                      hid_t dxpl_id __attribute__((unused)), void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *group;
    H5VL_pdc_obj_t *o          = (H5VL_pdc_obj_t *)obj;
    void *          under      = NULL;
    char *          group_name = (char *)calloc(1, strlen(name) + 1);
    strcpy(group_name, name);

    group             = H5VL_pdc_new_obj(under, o->under_vol_id);
    group->group_name = group_name;
    group->h5i_type   = H5I_GROUP;
    group->h5o_type   = H5O_TYPE_GROUP;

    char *file_name = (char *)calloc(1, strlen(o->file_name) + 1);
    strcpy(file_name, o->file_name);
    group->file_name = file_name;

    group->comm         = o->comm;
    group->info         = o->info;
    group->cont_id      = o->cont_id;
    group->file_obj_ptr = o->file_obj_ptr;

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return (void *)group;
} /* end H5VL_pdc_group_create() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_group_open(void *obj, const H5VL_loc_params_t *loc_params __attribute__((unused)), const char *name,
                    hid_t gapl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)), void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *group;
    H5VL_pdc_obj_t *o          = (H5VL_pdc_obj_t *)obj;
    void *          under      = NULL;
    char *          group_name = (char *)calloc(1, strlen(name) + 1);
    strcpy(group_name, name);

    group             = H5VL_pdc_new_obj(under, o->under_vol_id);
    group->group_name = group_name;
    group->h5i_type   = H5I_GROUP;
    group->h5o_type   = H5O_TYPE_GROUP;

    char *file_name = (char *)calloc(1, strlen(o->file_name) + 1);
    strcpy(file_name, o->file_name);
    group->file_name = file_name;

    group->comm         = o->comm;
    group->info         = o->info;
    group->cont_id      = o->cont_id;
    group->file_obj_ptr = o->file_obj_ptr;

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return (void *)group;
} /* end H5VL_pdc_group_open() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_group_get(void *obj __attribute__((unused)), H5VL_group_get_args_t *args __attribute__((unused)),
                   hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    return 0;
} /* end H5VL_pdc_group_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_group_specific(void *                      obj __attribute__((unused)),
                        H5VL_group_specific_args_t *args __attribute__((unused)),
                        hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_group_specific() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_group_optional(void *obj __attribute__((unused)), H5VL_optional_args_t *args __attribute__((unused)),
                        hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_group_optional() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_group_close(void *grp __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                     void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    return 0;
} /* end H5VL_pdc_group_close() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_link_create(H5VL_link_create_args_t *args __attribute__((unused)), void *obj __attribute__((unused)),
                     const H5VL_loc_params_t *loc_params __attribute__((unused)),
                     hid_t lcpl_id __attribute__((unused)), hid_t lapl_id __attribute__((unused)),
                     hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_link_create() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_link_copy(void *                   src_obj __attribute__((unused)),
                   const H5VL_loc_params_t *loc_params1 __attribute__((unused)),
                   void *                   dst_obj __attribute__((unused)),
                   const H5VL_loc_params_t *loc_params2 __attribute__((unused)),
                   hid_t lcpl_id __attribute__((unused)), hid_t lapl_id __attribute__((unused)),
                   hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_link_copy() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_link_move(void *                   src_obj __attribute__((unused)),
                   const H5VL_loc_params_t *loc_params1 __attribute__((unused)),
                   void *                   dst_obj __attribute__((unused)),
                   const H5VL_loc_params_t *loc_params2 __attribute__((unused)),
                   hid_t lcpl_id __attribute__((unused)), hid_t lapl_id __attribute__((unused)),
                   hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_link_move() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_link_get(void *                   obj __attribute__((unused)),
                  const H5VL_loc_params_t *loc_params __attribute__((unused)),
                  H5VL_link_get_args_t *args __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                  void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_link_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_link_specific(void *                     obj __attribute__((unused)),
                       const H5VL_loc_params_t *  loc_params __attribute__((unused)),
                       H5VL_link_specific_args_t *args __attribute__((unused)),
                       hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_link_specific() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_link_optional(void *                   obj __attribute__((unused)),
                       const H5VL_loc_params_t *loc_params __attribute__((unused)),
                       H5VL_optional_args_t *   args __attribute__((unused)),
                       hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_link_optional() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_attr_create(void *obj, const H5VL_loc_params_t *loc_params __attribute__((unused)), const char *name,
                     hid_t type_id, hid_t space_id, hid_t acpl_id __attribute__((unused)),
                     hid_t aapl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)), void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *attr;
    H5VL_pdc_obj_t *o     = (H5VL_pdc_obj_t *)obj;
    void *          under = NULL;
    psize_t         value_size;

    char *attr_name = (char *)malloc(strlen(name) + 1);
    strcpy(attr_name, name);
    attr                  = H5VL_pdc_new_obj(under, o->under_vol_id);
    attr->attr_name       = attr_name;
    value_size            = H5Sget_select_npoints(space_id) * H5Tget_size(type_id);
    attr->attr_value_size = value_size;
    attr->obj_id          = o->obj_id;
    attr->cont_id         = o->cont_id;

    attr->h5i_type = H5I_ATTR;
    /* attr->h5o_type = H5O_TYPE_ATTR; */

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return (void *)attr;
} /* end H5VL_pdc_attr_create() */
/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_attr_open(void *obj, const H5VL_loc_params_t *loc_params __attribute__((unused)), const char *name,
                   hid_t aapl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                   void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *attr;
    H5VL_pdc_obj_t *o         = (H5VL_pdc_obj_t *)obj;
    void *          under     = NULL;
    char *          attr_name = (char *)malloc(strlen(name) + 1);
    strcpy(attr_name, name);
    o->attr_name = attr_name;

    attr            = H5VL_pdc_new_obj(under, o->under_vol_id);
    attr->attr_name = attr_name;
    attr->obj_id    = o->obj_id;
    attr->cont_id   = o->cont_id;

    return (void *)attr;
} /* end H5VL_pdc_attr_open() */
/*---------------------------------------------------------------------------*/
static perr_t
H5VL_pdc_attr_read(void *attr, hid_t mem_type_id __attribute__((unused)), void *buf,
                   hid_t dxpl_id __attribute__((unused)), void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o         = (H5VL_pdc_obj_t *)attr;
    void *          tag_value = NULL;
    perr_t          ret_value = FAIL;
    pdc_var_type_t  value_type;

    if (o->obj_id > 0) {
        ret_value =
            PDCobj_get_tag(o->obj_id, (char *)o->attr_name, &tag_value, &value_type, &(o->attr_value_size));
    }
    else if (o->cont_id > 0) {
        ret_value =
            PDCcont_get_tag(o->cont_id, (char *)o->attr_name, &tag_value, &value_type, &(o->attr_value_size));
    }
    memcpy(buf, tag_value, o->attr_value_size);
    if (tag_value)
        free(tag_value);

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pdc_attr_read() */

/*---------------------------------------------------------------------------*/
static perr_t
H5VL_pdc_attr_write(void *attr, hid_t mem_type_id __attribute__((unused)), const void *buf,
                    hid_t dxpl_id __attribute__((unused)), void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *o         = (H5VL_pdc_obj_t *)attr;
    herr_t          ret_value = FAIL;

    if (o->obj_id > 0)
        ret_value =
            PDCobj_put_tag(o->obj_id, (char *)o->attr_name, (void *)buf, PDC_CHAR, o->attr_value_size);
    else if (o->cont_id > 0)
        ret_value =
            PDCcont_put_tag(o->cont_id, (char *)o->attr_name, (void *)buf, PDC_CHAR, o->attr_value_size);
    /* else */
    /*     HGOTO_ERROR(H5E_VOL, H5E_WRITEERROR, FAIL, "no valid PDC obj/cont ID"); */

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pdc_attr_write() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_attr_get(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id __attribute__((unused)), void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    FUNC_ENTER_VOL(herr_t, SUCCEED)

    H5VL_pdc_obj_t *o         = (H5VL_pdc_obj_t *)obj;
    void *          tag_value = NULL;
    pdc_var_type_t  value_type;

    if (o->obj_id > 0) {
        PDCobj_get_tag(o->obj_id, (char *)o->attr_name, &tag_value, &value_type, &(o->attr_value_size));
    }
    else if (o->cont_id > 0) {
        PDCcont_get_tag(o->cont_id, (char *)o->attr_name, &tag_value, &value_type, &(o->attr_value_size));
    }

    switch (args->op_type) {
        case H5VL_ATTR_GET_SPACE:
            hsize_t dims[1];
            dims[0] = o->attr_value_size / PDC_get_var_type_size(value_type);
            // TODO: handle compound
            hid_t space_id                = H5Screate_simple(1, dims, NULL);
            args->args.get_space.space_id = space_id;

            break;
        case H5VL_ATTR_GET_TYPE:
            hid_t dtype;
            switch (value_type) {
                case PDC_INT:
                    dtype = H5Tcopy(H5T_NATIVE_INT);
                    break;
                case PDC_FLOAT:
                    dtype = H5Tcopy(H5T_NATIVE_FLOAT);
                    break;
                case PDC_DOUBLE:
                    dtype = H5Tcopy(H5T_NATIVE_DOUBLE);
                    break;
                case PDC_CHAR:

                case PDC_STRING:
                    dtype = H5Tcopy(H5T_C_S1);
                    H5Tset_size(dtype, o->attr_value_size);
                    break;
                default:
                    fprintf(stderr, "Rank %d: %s unsupported PDC datatype type\n", my_rank_g, __func__);
                    HGOTO_ERROR(H5E_VOL, H5E_UNSUPPORTED, FAIL,
                                "invalid or unsupported attribute get operation");
            }
            args->args.get_type.type_id = dtype;

            break;
        default:
            fprintf(stderr, "Rank %d: %s unsupported get type\n", my_rank_g, __func__);
            HGOTO_ERROR(H5E_VOL, H5E_UNSUPPORTED, FAIL, "invalid or unsupported attribute get operation");
    }

    if (tag_value)
        free(tag_value);

done:
    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_attr_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_attr_specific(void *                     obj __attribute__((unused)),
                       const H5VL_loc_params_t *  loc_params __attribute__((unused)),
                       H5VL_attr_specific_args_t *args __attribute__((unused)),
                       hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_attr_specific() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_attr_optional(void *obj __attribute__((unused)), H5VL_optional_args_t *args __attribute__((unused)),
                       hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_attr_optional() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_attr_close(void *attr __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                    void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    /* H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)attr; */
    herr_t ret_value = SUCCEED;

    return ret_value;
} /* end H5VL_pdc_attr_close() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_object_open(void *obj, const H5VL_loc_params_t *loc_params, H5I_type_t *opened_type, hid_t dxpl_id,
                     void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *new_obj;
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    /* void *          under = NULL; */

    /* Only support dataset open and group open for now. */
    new_obj = H5VL_pdc_dataset_open(obj, loc_params, loc_params->loc_data.loc_by_name.name, 0, dxpl_id, req);
    if (new_obj == NULL) {
        new_obj =
            H5VL_pdc_group_open(obj, loc_params, loc_params->loc_data.loc_by_name.name, 0, dxpl_id, req);
        if (new_obj != NULL)
            *opened_type = H5I_GROUP;
    }
    else {
        *opened_type = H5I_DATASET;
    }

    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return (void *)new_obj;

    if (o->h5i_type == H5I_DATASET) {
        new_obj =
            H5VL_pdc_dataset_open(obj, loc_params, loc_params->loc_data.loc_by_name.name, 0, dxpl_id, req);
    }
    else if (loc_params->obj_type == H5I_GROUP) {
        new_obj =
            H5VL_pdc_group_open(obj, loc_params, loc_params->loc_data.loc_by_name.name, 0, dxpl_id, req);
    }
    else if (loc_params->obj_type == H5I_FILE) {
        new_obj = H5VL_pdc_attr_open(obj, loc_params, loc_params->loc_data.loc_by_name.name, 0, dxpl_id, req);
    }
    else {
        new_obj = NULL;
    }
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    return (void *)new_obj;
} /* end H5VL_pdc_object_open() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_object_copy(void *                   src_obj __attribute__((unused)),
                     const H5VL_loc_params_t *src_loc_params __attribute__((unused)),
                     const char *src_name __attribute__((unused)), void *dst_obj __attribute__((unused)),
                     const H5VL_loc_params_t *dst_loc_params __attribute__((unused)),
                     const char *dst_name __attribute__((unused)), hid_t ocpypl_id __attribute__((unused)),
                     hid_t lcpl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                     void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_object_copy() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_object_get(void *obj, const H5VL_loc_params_t *loc_params __attribute__((unused)),
                    H5VL_object_get_args_t *args, hid_t dxpl_id __attribute__((unused)), void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;

    switch (args->op_type) {
        case H5VL_OBJECT_GET_FILE:
            *(args->args.get_file.file) = o->file_obj_ptr;
            break;

        case H5VL_OBJECT_GET_NAME:
            HGOTO_ERROR(H5E_VOL, H5E_UNSUPPORTED, FAIL, "unsupported object get operation");
            break;

        case H5VL_OBJECT_GET_TYPE:
            *(args->args.get_type.obj_type) = o->h5o_type;
            break;

            /* case H5VL_OBJECT_GET_INFO: */
            /*     H5O_info2_t     *oinfo        = args->args.get_info.oinfo; */
            /*     unsigned         fields       = args->args.get_info.fields; */
            /*     break; */

        default:
            fprintf(stderr, "Rank %d: %s unsupported get type\n", my_rank_g, __func__);
            HGOTO_ERROR(H5E_VOL, H5E_UNSUPPORTED, FAIL, "invalid or unsupported object get operation");
    } /* end switch */

done:
    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, o->under_vol_id);

    FUNC_LEAVE_VOL
} /* end H5VL_pdc_object_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_object_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_specific_args_t *args,
                         hid_t dxpl_id, void **req)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;
    hid_t           under_vol_id;
    herr_t          ret_value;

    // Save copy of underlying VOL connector ID and prov helper, in case of
    // refresh destroying the current object
    under_vol_id = o->under_vol_id;

    ret_value = H5VLobject_specific(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if (req && *req)
        *req = H5VL_pdc_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_pdc_object_specific() */

static herr_t
H5VL_pdc_object_optional(void *                   obj __attribute__((unused)),
                         const H5VL_loc_params_t *loc_params __attribute__((unused)),
                         H5VL_optional_args_t *   args __attribute__((unused)),
                         hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_object_optional() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_request_wait(void *obj __attribute__((unused)), uint64_t timeout __attribute__((unused)),
                      H5VL_request_status_t *status __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_request_wait() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_request_notify(void *obj __attribute__((unused)), H5VL_request_notify_t cb __attribute__((unused)),
                        void *ctx __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_request_notify() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_request_cancel(void *                 obj __attribute__((unused)),
                        H5VL_request_status_t *status __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_request_cancel() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_request_specific(void *                        obj __attribute__((unused)),
                          H5VL_request_specific_args_t *args __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_request_specific() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_request_optional(void *                obj __attribute__((unused)),
                          H5VL_optional_args_t *args __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_request_optional() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_request_free(void *obj __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_request_free() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_blob_put(void *obj __attribute__((unused)), const void *buf __attribute__((unused)),
                  size_t size __attribute__((unused)), void *blob_id __attribute__((unused)),
                  void *ctx __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_blob_put() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_blob_get(void *obj __attribute__((unused)), const void *blob_id __attribute__((unused)),
                  void *buf __attribute__((unused)), size_t size __attribute__((unused)),
                  void *ctx __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_blob_get() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_blob_specific(void *obj __attribute__((unused)), void *blob_id __attribute__((unused)),
                       H5VL_blob_specific_args_t *args __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_blob_specific() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_blob_optional(void *obj __attribute__((unused)), void *blob_id __attribute__((unused)),
                       H5VL_optional_args_t *args __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_blob_optional() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_token_cmp(void *obj __attribute__((unused)), const H5O_token_t *token1 __attribute__((unused)),
                   const H5O_token_t *token2 __attribute__((unused)), int *cmp_value __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_token_cmp() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_token_to_str(void *obj __attribute__((unused)), H5I_type_t obj_type __attribute__((unused)),
                      const H5O_token_t *token __attribute__((unused)),
                      char **            token_str __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_token_to_str() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_token_from_str(void *obj __attribute__((unused)), H5I_type_t obj_type __attribute__((unused)),
                        const char * token_str __attribute__((unused)),
                        H5O_token_t *token __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_token_from_str() */

/*---------------------------------------------------------------------------*/
herr_t
H5VL_pdc_optional(void *obj __attribute__((unused)), H5VL_optional_args_t *args __attribute__((unused)),
                  hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    return 0;
} /* end H5VL_pdc_optional() */
