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
#include "H5Ldevelop.h"
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
    /* Hierarchy */
    char *   path;     /* group/dataset: path inside the file */
    char *   pdc_name; /* dataset: name of the PDC object */
    uint64_t meta_id;  /* global id of the container or object */
    /* File object elements */
    unsigned flags;
    hid_t    fapl_id;
    hid_t    fcpl_id;
    int      refcnt; /* open handle plus every open object of the file */
    /* Attribute object elements */
    hid_t                  attr_type;
    hid_t                  attr_space;
    struct H5VL_pdc_obj_t *attr_owner_tmp;
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

static void    _name_cache_free(void);
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

    _name_cache_free();

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
H5VL__pdc_file_init(const char *name, unsigned flags,
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
    file->flags        = flags;
    file->refcnt       = 1;
    file->num_procs    = 1;
    /* Copying an MPI-IO FAPL duplicates its communicator, which is collective, and a
     * read-only file may be open on a subset of the ranks. */
    file->fapl_id      = (flags & (H5F_ACC_RDWR | H5F_ACC_TRUNC | H5F_ACC_EXCL | H5F_ACC_CREAT))
                             ? H5Pcopy(fapl_id)
                             : H5Pcreate(H5P_FILE_ACCESS);
    file->path         = strdup("/");

    if (NULL == (file->file_name = strdup(name)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, NULL, "can't copy file name");

    driver = H5Pget_driver(fapl_id);
    /* H5Pget_fapl_mpio duplicates the communicator, which is collective, and a file
     * may be opened read-only on a subset of the ranks (CGNS does this on rank 0 to
     * detect the file type). Read-only files make no collective changes. */
    if (driver == H5FD_MPIO && (flags & (H5F_ACC_RDWR | H5F_ACC_TRUNC | H5F_ACC_EXCL | H5F_ACC_CREAT))) {
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
static herr_t
H5VL__pdc_file_close(H5VL_pdc_obj_t *file)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(file);

    /* Objects of the file may outlive H5Fclose */
    if (--file->refcnt > 0)
        HGOTO_DONE(SUCCEED);

    if (file->req_alloc) {
        free(file->xfer_requests);
        free(file->bufs);
        file->req_alloc = 0;
    }

    if (PDCcont_close(file->cont_id) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, FAIL, "failed to close container");

    /* Free file data structures */
    if (file->file_name)
        free(file->file_name);
    free(file->path);
    if (file->fapl_id > 0)
        H5Pclose(file->fapl_id);
    if (file->fcpl_id > 0)
        H5Pclose(file->fcpl_id);
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
    dset->file_obj_ptr->refcnt++;

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
/* Hierarchy support.
 *
 *   file      -> PDC container named by the file path
 *   group     -> PDC container nested under its parent container
 *   dataset   -> PDC object listed in its parent container. PDC object names are
 *                global, so the name carries the full location:
 *                <leaf>/<parent path>/<file path>
 *   attribute -> PDC tag on the object or container, described by an index tag
 *                (see H5VL_PDC_ATTR_INDEX)
 *
 *   soft or external link -> an empty container whose H5VL_PDC_LINK_TAG tag holds the
 *                target. Lookups follow it; a dangling link only fails when followed.
 *
 * Link creation order is the order of the container's entries. Lookups are
 * independent, because HDF5 and CGNS may read metadata on a subset of ranks (CGNS
 * opens a file on rank 0 alone to detect its type). Changes are collective: rank 0
 * makes them and the others wait for its status.
 */
#define H5VL_PDC_ATTR_INDEX "__h5vol_attr_index"
#define H5VL_PDC_LINK_TAG   "__h5vol_link"
#define H5VL_PDC_MAX_LINKS  16 /* depth of links followed, which also stops loops */

typedef struct _pdc_loc_t {
    H5VL_pdc_obj_t *  file; /* holds a reference; a link can lead to another file */
    pdc_entity_type_t type;
    uint64_t          meta_id;
    char *            path;     /* path inside the file */
    char *            pdc_name; /* name PDC knows the entry by */
} _pdc_loc_t;

static herr_t H5VL_pdc_group_close(void *grp, hid_t dxpl_id, void **req);
static herr_t _flush_xfer_requests(H5VL_pdc_obj_t *file);
static herr_t H5VL__pdc_file_close(H5VL_pdc_obj_t *file);

/* Name cache. A container's or object's name never changes for a given id and ids
 * are not reused, so a name needs fetching once. Listings still come from the server,
 * so what a group contains is always current. */
typedef struct _name_slot_t {
    uint64_t id;
    char *   name;
    /* Whether the id is a link, also fixed for its lifetime */
    int        link_known;
    H5L_type_t link_type;
    char *     link_path;
    char *     link_file;
} _name_slot_t;

static _name_slot_t *name_cache_g      = NULL;
static size_t        name_cache_cap_g  = 0;
static size_t        name_cache_used_g = 0;

static size_t
_name_hash(uint64_t id)
{
    id ^= id >> 33;
    id *= 0xff51afd7ed558ccdULL;
    id ^= id >> 33;
    return (size_t)id;
}

static void
_name_cache_insert(uint64_t id, char *name)
{
    size_t i;
    if ((name_cache_used_g + 1) * 2 > name_cache_cap_g) {
        size_t        cap = name_cache_cap_g ? name_cache_cap_g * 2 : 1024;
        _name_slot_t *t   = (_name_slot_t *)calloc(cap, sizeof(_name_slot_t));
        for (size_t k = 0; k < name_cache_cap_g; k++) {
            if (name_cache_g[k].id) {
                i = _name_hash(name_cache_g[k].id) & (cap - 1);
                while (t[i].id)
                    i = (i + 1) & (cap - 1);
                t[i] = name_cache_g[k];
            }
        }
        free(name_cache_g);
        name_cache_g     = t;
        name_cache_cap_g = cap;
    }
    i = _name_hash(id) & (name_cache_cap_g - 1);
    while (name_cache_g[i].id)
        i = (i + 1) & (name_cache_cap_g - 1);
    memset(&name_cache_g[i], 0, sizeof(name_cache_g[i]));
    name_cache_g[i].id   = id;
    name_cache_g[i].name = name;
    name_cache_used_g++;
}

/* The name PDC holds for an id, or NULL for a dead entry. Owned by the cache. */
static const char *
_entry_name(uint64_t id)
{
    char *name = NULL;
    if (name_cache_cap_g) {
        size_t i = _name_hash(id) & (name_cache_cap_g - 1);
        while (name_cache_g[i].id) {
            if (name_cache_g[i].id == id)
                return name_cache_g[i].name;
            i = (i + 1) & (name_cache_cap_g - 1);
        }
    }
    if (PDCcont_get((pdcid_t)id, &name) != SUCCEED || name == NULL) {
        free(name);
        return NULL;
    }
    _name_cache_insert(id, name);
    return name;
}

static void
_name_cache_free(void)
{
    for (size_t k = 0; k < name_cache_cap_g; k++) {
        free(name_cache_g[k].name);
        free(name_cache_g[k].link_path);
        free(name_cache_g[k].link_file);
    }
    free(name_cache_g);
    name_cache_g      = NULL;
    name_cache_cap_g  = 0;
    name_cache_used_g = 0;
}

static H5VL_pdc_obj_t *H5VL__pdc_file_init(const char *name, unsigned flags, H5VL_pdc_info_t *info,
                                           hid_t fapl_id);

static H5VL_pdc_obj_t *
_file_of(H5VL_pdc_obj_t *o)
{
    return o->file_obj_ptr ? o->file_obj_ptr : o;
}

static int
_is_coll(H5VL_pdc_obj_t *file)
{
    return file->comm != MPI_COMM_NULL;
}

static int
_is_writer(H5VL_pdc_obj_t *file)
{
    return !_is_coll(file) || file->my_rank == 0;
}

/* A collective change. Lookups are independent and PDC has no transaction: a rank
 * still checking whether a name exists (as CGNS does before every create) must not
 * see rank 0's change, or it reports a duplicate. Every rank's earlier lookups are
 * finished once all have arrived here; rank 0 then changes and the others wait in
 * _sync_status. */
static void
_change_begin(H5VL_pdc_obj_t *file)
{
    if (_is_coll(file))
        MPI_Barrier(file->comm);
}

/* All ranks of a collective change learn rank 0's result; this also orders them
 * after rank 0's RPCs have completed. */
static herr_t
_sync_status(H5VL_pdc_obj_t *file, herr_t status)
{
    int bad = status < 0;
    if (_is_coll(file))
        MPI_Bcast(&bad, 1, MPI_INT, 0, file->comm);
    return bad ? -1 : 0;
}

static char *
_path_join(const char *parent, const char *leaf)
{
    size_t n = strlen(parent) + strlen(leaf) + 2;
    char * p = (char *)malloc(n);
    snprintf(p, n, "%s/%s", parent, leaf);
    replace_multi_slash(p);
    return p;
}

/* PDC name of a dataset. One name per HDF5 path: a leaf has no slash in it. */
static char *
_pdc_obj_name(H5VL_pdc_obj_t *file, const char *parent_path, const char *leaf)
{
    size_t n = strlen(leaf) + strlen(parent_path) + strlen(file->file_name) + 3;
    char * p = (char *)malloc(n);
    snprintf(p, n, "%s/%s/%s", leaf, parent_path, file->file_name);
    replace_multi_slash(p);
    return p;
}

/* Link name of an entry from the name PDC holds for it */
static char *
_leaf_of(const char *pdc_name)
{
    const char *slash = strchr(pdc_name, '/');
    size_t      n     = slash ? (size_t)(slash - pdc_name) : strlen(pdc_name);
    char *      p     = (char *)malloc(n + 1);
    memcpy(p, pdc_name, n);
    p[n] = '\0';
    return p;
}

static void
_loc_free(_pdc_loc_t *loc)
{
    free(loc->path);
    free(loc->pdc_name);
    loc->path = loc->pdc_name = NULL;
    if (loc->file)
        H5VL__pdc_file_close(loc->file);
    loc->file = NULL;
}

/* 0 found, 1 not found, -1 error. cont is a local handle or a meta id. */
static int
_find_child(pdcid_t cont, const char *leaf, uint64_t *meta_id, pdc_entity_type_t *type, char **pdc_name,
            hsize_t *position)
{
    int               n = 0, ret = 1;
    pdc_cont_entry_t *entries = NULL;

    if (PDCcont_get_entries(cont, &n, &entries) != SUCCEED)
        return -1;
    for (int i = 0; i < n && ret == 1; i++) {
        const char *name = _entry_name(entries[i].meta_id);
        if (name) {
            char *l = _leaf_of(name);
            if (strcmp(l, leaf) == 0) {
                ret = 0;
                if (meta_id)
                    *meta_id = entries[i].meta_id;
                if (type)
                    *type = entries[i].type;
                if (position)
                    *position = (hsize_t)i;
                if (pdc_name)
                    *pdc_name = strdup(name);
            }
            free(l);
        }
    }
    free(entries);
    return ret;
}

/* The target of a link container: 1 yes, 0 not a link. Strings are owned by the cache. */
static int
_entry_link(uint64_t id, H5L_type_t *type, const char **path, const char **file)
{
    _name_slot_t *slot = NULL;
    void *        tag = NULL;
    psize_t       size = 0;
    pdc_var_type_t tag_type;
    pdcid_t       h;

    if (_entry_name(id) == NULL)
        return 0;
    for (size_t i = _name_hash(id) & (name_cache_cap_g - 1); name_cache_g[i].id; i = (i + 1) & (name_cache_cap_g - 1)) {
        if (name_cache_g[i].id == id) {
            slot = &name_cache_g[i];
            break;
        }
    }
    if (slot == NULL)
        return 0;
    if (!slot->link_known) {
        if ((h = PDCcont_open_id((pdcid_t)id, pdc_id_g)) == 0)
            return 0;
        if (PDCcont_get_tag(h, H5VL_PDC_LINK_TAG, &tag, &tag_type, &size) == SUCCEED && tag && size > 1) {
            /* type byte, path, NUL, file, NUL */
            const char *blob = (const char *)tag;
            slot->link_type = blob[0] == 'E' ? H5L_TYPE_EXTERNAL : H5L_TYPE_SOFT;
            slot->link_path = strdup(blob + 1);
            slot->link_file = strdup(blob + 2 + strlen(blob + 1));
        }
        free(tag);
        PDCcont_close(h);
        slot->link_known = 1;
    }
    if (slot->link_path == NULL)
        return 0;
    if (type)
        *type = slot->link_type;
    if (path)
        *path = slot->link_path;
    if (file)
        *file = slot->link_file;
    return 1;
}

/* A file reached through an external link, open only for the objects inside it.
 * The target is relative to the directory of the linking file. Returns with one
 * reference held by the caller. Such a file is read-only and independent. */
static H5VL_pdc_obj_t *
_open_external(H5VL_pdc_obj_t *from, const char *target)
{
    H5VL_pdc_obj_t *file;
    const char *    slash = strrchr(from->file_name, '/');
    size_t          dir   = slash ? (size_t)(slash - from->file_name) + 1 : 0;
    char *          name;
    pdcid_t         cont;

    while (target[0] == '.' && target[1] == '/')
        target += 2;
    if (target[0] == '/')
        dir = 0;
    name = (char *)malloc(dir + strlen(target) + 1);
    memcpy(name, from->file_name, dir);
    strcpy(name + dir, target);
    cont = PDCcont_open(name, pdc_id_g);
    if (cont == 0) {
        free(name);
        return NULL;
    }
    file = H5VL__pdc_file_init(name, H5F_ACC_RDONLY, NULL, H5P_FILE_ACCESS_DEFAULT);
    if (file) {
        file->cont_id = cont;
        file->meta_id = PDCcont_get_info(name)->meta_id;
        file->fcpl_id = H5Pcreate(H5P_FILE_CREATE);
    }
    free(name);
    return file;
}

/* Follow name from a starting location. 0 found, 1 not found, -1 error. follow_last
 * says whether to go through a link at the end of the path: H5Lexists and link
 * queries are about the link itself, opens are about its target. */
static int
_resolve_from(H5VL_pdc_obj_t *file, pdc_entity_type_t type, uint64_t meta_id, const char *path,
              const char *pdc_name, const char *name, int follow_last, int depth, _pdc_loc_t *loc)
{
    int   ret = 0;
    char *copy, *comp, *next, *save = NULL;

    memset(loc, 0, sizeof(*loc));
    file->refcnt++;
    loc->file     = file;
    loc->type     = type;
    loc->meta_id  = meta_id;
    loc->path     = strdup(path);
    loc->pdc_name = pdc_name ? strdup(pdc_name) : NULL;

    copy = strdup(name);
    comp = strtok_r(copy, "/", &save);
    while (comp && strcmp(comp, ".") == 0)
        comp = strtok_r(NULL, "/", &save);
    for (; comp && ret == 0; comp = next) {
        char *            child_name = NULL;
        uint64_t          child_id;
        pdc_entity_type_t child_type;
        H5L_type_t        link_type;
        const char *      link_path, *link_file;

        next = strtok_r(NULL, "/", &save);
        while (next && strcmp(next, ".") == 0)
            next = strtok_r(NULL, "/", &save);
        if (loc->type != PDC_ENTITY_CONT) {
            ret = 1;
            break;
        }
        ret = _find_child((pdcid_t)loc->meta_id, comp, &child_id, &child_type, &child_name, NULL);
        if (ret != 0)
            break;
        if (child_type == PDC_ENTITY_CONT && (next || follow_last) &&
            _entry_link(child_id, &link_type, &link_path, &link_file)) {
            H5VL_pdc_obj_t *target = loc->file;
            _pdc_loc_t      sub;

            free(child_name);
            if (depth >= H5VL_PDC_MAX_LINKS) {
                ret = 1;
                break;
            }
            if (link_type == H5L_TYPE_EXTERNAL && NULL == (target = _open_external(loc->file, link_file))) {
                ret = 1;
                break;
            }
            if (link_type == H5L_TYPE_SOFT && link_path[0] != '/')
                ret = _resolve_from(target, PDC_ENTITY_CONT, loc->meta_id, loc->path, NULL, link_path, 1,
                                    depth + 1, &sub);
            else
                ret = _resolve_from(target, PDC_ENTITY_CONT, target->meta_id, "/", NULL, link_path, 1, depth + 1,
                                    &sub);
            if (link_type == H5L_TYPE_EXTERNAL)
                H5VL__pdc_file_close(target); /* sub holds its own reference */
            _loc_free(loc);
            *loc = sub;
        }
        else {
            char *child_path = _path_join(loc->path, comp);
            H5VL_pdc_obj_t *f = loc->file;
            f->refcnt++;
            _loc_free(loc);
            loc->file     = f;
            loc->meta_id  = child_id;
            loc->type     = child_type;
            loc->path     = child_path;
            loc->pdc_name = child_name;
        }
    }
    free(copy);
    if (ret != 0)
        _loc_free(loc);
    return ret;
}

static int
_resolve_o(H5VL_pdc_obj_t *o, const char *name, int follow_last, _pdc_loc_t *loc)
{
    H5VL_pdc_obj_t *file = _file_of(o);

    if (name[0] == '/')
        return _resolve_from(file, PDC_ENTITY_CONT, file->meta_id, "/", NULL, name, follow_last, 0, loc);
    return _resolve_from(file, o->h5i_type == H5I_DATASET ? PDC_ENTITY_OBJ : PDC_ENTITY_CONT, o->meta_id,
                         o->path ? o->path : "/", o->h5i_type == H5I_DATASET ? o->pdc_name : NULL, name,
                         follow_last, 0, loc);
}

static int
_resolve(H5VL_pdc_obj_t *o, const char *name, _pdc_loc_t *loc)
{
    return _resolve_o(o, name, 1, loc);
}

/* Split "a/b/c" into parent "a/b" and leaf "c". parent is "." when name has one component. */
static void
_split_name(const char *name, char **parent, char **leaf)
{
    char * copy = strdup(name);
    size_t n    = strlen(copy);
    char * slash;

    while (n > 1 && copy[n - 1] == '/')
        copy[--n] = '\0';
    slash = strrchr(copy, '/');
    if (!slash) {
        *parent = strdup(".");
        *leaf   = strdup(copy);
    }
    else {
        *leaf = strdup(slash + 1);
        if (slash == copy)
            *parent = strdup("/");
        else {
            *slash  = '\0';
            *parent = strdup(copy);
        }
    }
    free(copy);
}

static H5VL_pdc_obj_t *
_open_group_obj(H5VL_pdc_obj_t *file, uint64_t meta_id, const char *path, pdcid_t handle)
{
    H5VL_pdc_obj_t *group = H5VL_pdc_new_obj(NULL, file->under_vol_id);

    group->cont_id = handle ? handle : PDCcont_open_id((pdcid_t)meta_id, pdc_id_g);
    if (group->cont_id == 0) {
        free(group);
        return NULL;
    }
    group->under_object = group;
    group->meta_id      = meta_id;
    group->path         = strdup(path);
    group->group_name   = strdup(path);
    group->file_name    = strdup(file->file_name);
    group->h5i_type     = H5I_GROUP;
    group->h5o_type     = H5O_TYPE_GROUP;
    group->comm         = file->comm;
    group->info         = file->info;
    group->my_rank      = file->my_rank;
    group->num_procs    = file->num_procs;
    group->file_obj_ptr = file;
    file->refcnt++;
    return group;
}

static H5VL_pdc_obj_t *
_open_dset_obj(H5VL_pdc_obj_t *file, const _pdc_loc_t *loc)
{
    H5VL_pdc_obj_t *     dset = NULL;
    struct pdc_obj_info *info;
    char *               leaf;

    if (NULL == (dset = H5VL__pdc_dset_init(file)))
        return NULL;
    leaf = _leaf_of(loc->pdc_name);
    strncpy(dset->obj_name, leaf, sizeof(dset->obj_name) - 1);
    free(leaf);
    dset->obj_id = PDCobj_open(loc->pdc_name, pdc_id_g);
    if (dset->obj_id <= 0) {
        file->refcnt--;
        free(dset);
        return NULL;
    }
    dset->meta_id      = loc->meta_id;
    dset->path         = strdup(loc->path);
    dset->pdc_name     = strdup(loc->pdc_name);
    dset->under_vol_id = file->under_vol_id;
    dset->under_object = dset;
    info               = PDCobj_get_info(dset->obj_id);
    dset->pdc_type     = info->obj_pt->type;

    // TODO: temporary workaround for writing compound data, as current PDC doesn't support
    //       compound datatype
    if (dset->pdc_type == PDC_CHAR) {
        psize_t        value_size = 0;
        pdc_var_type_t value_type;
        psize_t *      value = NULL;
        PDCobj_get_tag(dset->obj_id, "PDC_COMPOUND_DTYPE_SIZE", (void **)&value, &value_type, &value_size);
        if (value && value_size > 0) {
            dset->compound_size = *value;
            info->obj_pt->dims[info->obj_pt->ndim - 1] /= *value;
        }
    }
    dset->space_id = H5Screate_simple(info->obj_pt->ndim, info->obj_pt->dims, NULL);
    dset->type_id  = _pdc_to_h5_type(dset->pdc_type);
    dset->dcpl_id  = H5Pcreate(H5P_DATASET_CREATE);
    return dset;
}

static H5VL_pdc_obj_t *
_open_loc(H5VL_pdc_obj_t *file, const _pdc_loc_t *loc)
{
    if (loc->type == PDC_ENTITY_CONT)
        return _open_group_obj(file, loc->meta_id, loc->path, 0);
    return _open_dset_obj(file, loc);
}

static void
_close_obj(H5VL_pdc_obj_t *o)
{
    if (o->h5i_type == H5I_DATASET)
        H5VL_pdc_dataset_close(o, H5P_DEFAULT, NULL);
    else
        H5VL_pdc_group_close(o, H5P_DEFAULT, NULL);
}

/* The object a loc_params names. *tmp is set when it was opened here and must be closed. */
static H5VL_pdc_obj_t *
_loc_owner(H5VL_pdc_obj_t *o, const H5VL_loc_params_t *loc_params, H5VL_pdc_obj_t **tmp)
{
    _pdc_loc_t      loc;
    H5VL_pdc_obj_t *res;

    *tmp = NULL;
    if (loc_params == NULL || loc_params->type == H5VL_OBJECT_BY_SELF)
        return o;
    if (loc_params->type != H5VL_OBJECT_BY_NAME)
        return NULL;
    if (strcmp(loc_params->loc_data.loc_by_name.name, ".") == 0)
        return o;
    if (_resolve(o, loc_params->loc_data.loc_by_name.name, &loc) != 0)
        return NULL;
    res = _open_loc(loc.file, &loc);
    _loc_free(&loc);
    *tmp = res;
    return res;
}

static void
_token_of(uint64_t meta_id, H5O_token_t *token)
{
    memset(token, 0, sizeof(*token));
    memcpy(token, &meta_id, sizeof(meta_id));
}

/* Tags: PDC keeps the first value when a name is put twice, so replace explicitly */
static herr_t
_tag_get(H5VL_pdc_obj_t *o, const char *name, void **value, psize_t *size, pdc_var_type_t *type)
{
    pdc_var_type_t t;
    *value = NULL;
    *size  = 0;
    if (o->obj_id > 0)
        return PDCobj_get_tag(o->obj_id, (char *)name, value, type ? type : &t, size) == SUCCEED ? 0 : -1;
    return PDCcont_get_tag(o->cont_id, (char *)name, value, type ? type : &t, size) == SUCCEED ? 0 : -1;
}

static herr_t
_tag_del(H5VL_pdc_obj_t *o, const char *name)
{
    void *  old = NULL;
    psize_t size;
    if (_tag_get(o, name, &old, &size, NULL) < 0 || old == NULL)
        return 0;
    free(old);
    if (o->obj_id > 0)
        return PDCobj_del_tag(o->obj_id, (char *)name) == SUCCEED ? 0 : -1;
    return PDCcont_del_tag(o->cont_id, (char *)name) == SUCCEED ? 0 : -1;
}

static herr_t
_tag_put(H5VL_pdc_obj_t *o, const char *name, const void *value, psize_t size)
{
    if (_tag_del(o, name) < 0)
        return -1;
    if (o->obj_id > 0)
        return PDCobj_put_tag(o->obj_id, (char *)name, (void *)value, PDC_CHAR, size) == SUCCEED ? 0 : -1;
    return PDCcont_put_tag(o->cont_id, (char *)name, (void *)value, PDC_CHAR, size) == SUCCEED ? 0 : -1;
}

/* Attribute index: for each attribute, in creation order,
 *   uint32 name_len, uint32 type_len, uint32 space_len, name, H5Tencode, H5Sencode
 * It records what a PDC tag cannot: the names, the datatype and the dataspace. */
typedef struct _attr_rec_t {
    const char *         name;
    uint32_t             name_len;
    const unsigned char *type_buf;
    uint32_t             type_len;
    const unsigned char *space_buf;
    uint32_t             space_len;
} _attr_rec_t;

static const unsigned char *
_attr_next(const unsigned char *p, const unsigned char *end, _attr_rec_t *rec)
{
    uint32_t h[3];
    if (p == NULL || (size_t)(end - p) < sizeof(h))
        return NULL;
    memcpy(h, p, sizeof(h));
    p += sizeof(h);
    rec->name      = (const char *)p;
    rec->name_len  = h[0];
    rec->type_buf  = p + h[0];
    rec->type_len  = h[1];
    rec->space_buf = rec->type_buf + h[1];
    rec->space_len = h[2];
    return rec->space_buf + h[2];
}

static int
_attr_find(const unsigned char *buf, psize_t len, const char *name, _attr_rec_t *out, int *position)
{
    const unsigned char *p = buf, *end = buf ? buf + len : NULL;
    _attr_rec_t          rec;
    int                  i = 0;
    while ((p = _attr_next(p, end, &rec)) != NULL) {
        if (strlen(name) == rec.name_len && memcmp(rec.name, name, rec.name_len) == 0) {
            if (out)
                *out = rec;
            if (position)
                *position = i;
            return 1;
        }
        i++;
    }
    return 0;
}

static herr_t
_attr_index_add(H5VL_pdc_obj_t *owner, const char *name, hid_t type_id, hid_t space_id)
{
    void *         old = NULL;
    psize_t        old_len;
    size_t         type_len = 0, space_len = 0, add_len, name_len = strlen(name);
    unsigned char *buf, *p;
    uint32_t       h[3];
    herr_t         ret;

    if (_tag_get(owner, H5VL_PDC_ATTR_INDEX, &old, &old_len, NULL) < 0)
        return -1;
    if (old == NULL)
        old_len = 0;
    if (H5Tencode(type_id, NULL, &type_len) < 0 || H5Sencode2(space_id, NULL, &space_len, H5P_DEFAULT) < 0)
        return -1;
    add_len = sizeof(h) + name_len + type_len + space_len;
    buf     = (unsigned char *)malloc(old_len + add_len);
    if (old_len)
        memcpy(buf, old, old_len);
    p    = buf + old_len;
    h[0] = (uint32_t)name_len;
    h[1] = (uint32_t)type_len;
    h[2] = (uint32_t)space_len;
    memcpy(p, h, sizeof(h));
    p += sizeof(h);
    memcpy(p, name, name_len);
    p += name_len;
    H5Tencode(type_id, p, &type_len);
    p += type_len;
    H5Sencode2(space_id, p, &space_len, H5P_DEFAULT);
    ret = _tag_put(owner, H5VL_PDC_ATTR_INDEX, buf, old_len + add_len);
    free(buf);
    free(old);
    return ret;
}

static herr_t
_attr_index_remove(H5VL_pdc_obj_t *owner, const char *name)
{
    void *      old = NULL;
    psize_t     old_len;
    _attr_rec_t rec;
    herr_t      ret;

    if (_tag_get(owner, H5VL_PDC_ATTR_INDEX, &old, &old_len, NULL) < 0)
        return -1;
    if (old == NULL || !_attr_find(old, old_len, name, &rec, NULL)) {
        free(old);
        return -1;
    }
    {
        size_t         rec_start = (const unsigned char *)rec.name - (const unsigned char *)old - 3 * sizeof(uint32_t);
        size_t         rec_len   = 3 * sizeof(uint32_t) + rec.name_len + rec.type_len + rec.space_len;
        size_t         new_len   = old_len - rec_len;
        unsigned char *buf       = (unsigned char *)malloc(new_len ? new_len : 1);
        memcpy(buf, old, rec_start);
        memcpy(buf + rec_start, (unsigned char *)old + rec_start + rec_len, new_len - rec_start);
        if (new_len)
            ret = _tag_put(owner, H5VL_PDC_ATTR_INDEX, buf, new_len);
        else
            ret = _tag_del(owner, H5VL_PDC_ATTR_INDEX);
        free(buf);
    }
    free(old);
    return ret < 0 ? -1 : _tag_del(owner, name);
}

/* Remove an entry and everything under it. Caller is the writing rank. */
static herr_t
_delete_entry(pdcid_t parent, uint64_t meta_id, pdc_entity_type_t type, const char *pdc_name)
{
    pdcid_t           id = (pdcid_t)meta_id;
    pdc_entity_type_t t  = type;
    herr_t            ret = 0;

    if (PDCcont_del_entries(parent, 1, &id, &t) != SUCCEED)
        ret = -1;
    if (type == PDC_ENTITY_OBJ) {
        pdcid_t h = PDCobj_open(pdc_name, pdc_id_g);
        if (h <= 0 || PDCobj_del(h) != SUCCEED)
            ret = -1;
        if (h > 0)
            PDCobj_close(h);
    }
    else {
        pdcid_t           h = PDCcont_open_id(id, pdc_id_g);
        int               n = 0;
        pdc_cont_entry_t *entries = NULL;
        if (h == 0)
            return -1;
        if (PDCcont_get_entries(h, &n, &entries) == SUCCEED) {
            for (int i = 0; i < n; i++) {
                const char *name = _entry_name(entries[i].meta_id);
                if (name)
                    ret |= _delete_entry(h, entries[i].meta_id, entries[i].type, name);
            }
            free(entries);
        }
        if (PDCcont_del(h) != SUCCEED)
            ret = -1;
        PDCcont_close(h);
    }
    return ret;
}

/* H5F_ACC_TRUNC on a container that already exists: start empty again */
static herr_t
_truncate_root(H5VL_pdc_obj_t *file)
{
    int               n = 0;
    pdc_cont_entry_t *entries = NULL;
    void *            idx = NULL;
    psize_t           idx_len;
    herr_t            ret = 0;

    if (PDCcont_get_entries(file->cont_id, &n, &entries) != SUCCEED)
        return -1;
    for (int i = 0; i < n; i++) {
        const char *name = _entry_name(entries[i].meta_id);
        if (name)
            ret |= _delete_entry(file->cont_id, entries[i].meta_id, entries[i].type, name);
    }
    free(entries);
    if (_tag_get(file, H5VL_PDC_ATTR_INDEX, &idx, &idx_len, NULL) == 0 && idx) {
        const unsigned char *p = idx, *end = p + idx_len;
        _attr_rec_t          rec;
        while ((p = _attr_next(p, end, &rec)) != NULL) {
            char *name = strndup(rec.name, rec.name_len);
            ret |= _tag_del(file, name);
            free(name);
        }
        free(idx);
        ret |= _tag_del(file, H5VL_PDC_ATTR_INDEX);
    }
    return ret;
}

/* Count or list the open objects of a file (H5Fget_obj_count / H5Fget_obj_ids) */
typedef struct _obj_count_t {
    H5VL_pdc_obj_t *file;
    H5I_type_t      type;
    size_t          max;
    hid_t *         ids;
    size_t          count;
} _obj_count_t;

static herr_t
_obj_count_cb(hid_t id, void *data)
{
    _obj_count_t *c = (_obj_count_t *)data;
    hbool_t       native = TRUE;
    void *        obj;

    if (H5VLobject_is_native(id, &native) < 0 || native)
        return 0;
    if (NULL == (obj = H5VLobject(id)))
        return 0;
    if (c->type == H5I_FILE ? obj != (void *)c->file : ((H5VL_pdc_obj_t *)obj)->file_obj_ptr != c->file)
        return 0;
    if (c->ids && c->count < c->max)
        c->ids[c->count] = id;
    c->count++;
    return 0;
}

static size_t
_obj_count(H5VL_pdc_obj_t *file, unsigned types, size_t max, hid_t *ids)
{
    static const struct {
        unsigned   flag;
        H5I_type_t type;
    } map[] = {{H5F_OBJ_FILE, H5I_FILE}, {H5F_OBJ_DATASET, H5I_DATASET}, {H5F_OBJ_GROUP, H5I_GROUP},
               {H5F_OBJ_ATTR, H5I_ATTR}};
    _obj_count_t c = {file, H5I_FILE, max, ids, 0};

    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
        if (types & map[i].flag) {
            c.type = map[i].type;
            H5Iiterate(map[i].type, _obj_count_cb, &c);
        }
    }
    return c.count;
}

static void
_fill_oinfo(H5VL_pdc_obj_t *file, uint64_t meta_id, H5O_type_t type, hsize_t num_attrs, H5O_info2_t *oinfo)
{
    memset(oinfo, 0, sizeof(*oinfo));
    oinfo->fileno    = file->meta_id;
    oinfo->type      = type;
    oinfo->rc        = 1;
    oinfo->num_attrs = num_attrs;
    _token_of(meta_id, &oinfo->token);
}

static hsize_t
_attr_count(H5VL_pdc_obj_t *owner)
{
    void *               idx = NULL;
    psize_t              len;
    hsize_t              count = 0;
    const unsigned char *p, *end;
    _attr_rec_t          rec;

    if (_tag_get(owner, H5VL_PDC_ATTR_INDEX, &idx, &len, NULL) < 0 || idx == NULL)
        return 0;
    p   = idx;
    end = p + len;
    while ((p = _attr_next(p, end, &rec)) != NULL)
        count++;
    free(idx);
    return count;
}

/*---------------------------------------------------------------------------*/
static herr_t
H5VL__pdc_dset_free(H5VL_pdc_obj_t *dset)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    if (dset->dcpl_id > 0)
        H5Pclose(dset->dcpl_id);
    if (dset->dapl_id > 0)
        H5Pclose(dset->dapl_id);
    if (dset->dxpl_id > 0)
        H5Pclose(dset->dxpl_id);
    free(dset->path);
    free(dset->pdc_name);
    if (dset->type_id != 0)
        H5Tclose(dset->type_id);
    if (dset->space_id != 0 && dset->space_id != H5S_ALL)
        H5Sclose(dset->space_id);

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
H5VL_pdc_file_create(const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id,
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
    file->meta_id = PDCcont_get_info(name)->meta_id;
    file->fcpl_id = H5Pcopy(fcpl_id);

    /* PDC returns the existing container if the name is taken */
    {
        herr_t status = 0;
        if (_is_writer(file))
            status = _truncate_root(file);
        if (_sync_status(file, status) < 0)
            HGOTO_ERROR(H5E_FILE, H5E_CANTCREATE, NULL, "can't truncate existing container");
    }

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
    file->meta_id = PDCcont_get_info(name)->meta_id;
    file->fcpl_id = H5Pcreate(H5P_FILE_CREATE);

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

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    assert(file);

    // Complete existing write requests
    if (_flush_xfer_requests(file) != SUCCEED)
        HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "Failed to complete deferred writes");

    if (H5VL__pdc_file_close(file) < 0)
        HGOTO_ERROR(H5E_FILE, H5E_CANTCLOSEFILE, FAIL, "failed to close file");

done:
    FUNC_LEAVE_VOL
} /* end H5VL_pdc_file_close() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_file_specific(void *file, H5VL_file_specific_args_t *args, hid_t dxpl_id __attribute__((unused)),
                       void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)file;

    switch (args->op_type) {
        case H5VL_FILE_FLUSH:
            /* the deferred write queue lives on the file object */
            return _flush_xfer_requests(_file_of(o)) == SUCCEED ? 0 : -1;
        case H5VL_FILE_IS_ACCESSIBLE: {
            pdcid_t cont = PDCcont_open(args->args.is_accessible.filename, pdc_id_g);
            *args->args.is_accessible.accessible = cont > 0;
            if (cont > 0)
                PDCcont_close(cont);
            return 0;
        }
        case H5VL_FILE_IS_EQUAL:
            *args->args.is_equal.same_file =
                _file_of(o)->meta_id == _file_of((H5VL_pdc_obj_t *)args->args.is_equal.obj2)->meta_id;
            return 0;
        case H5VL_FILE_REOPEN:
        case H5VL_FILE_DELETE:
        default:
            return -1;
    }
} /* end H5VL_pdc_file_specific() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_file_get(void *_file, H5VL_file_get_args_t *args, hid_t dxpl_id __attribute__((unused)),
                  void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *file = _file_of((H5VL_pdc_obj_t *)_file);

    switch (args->op_type) {
        case H5VL_FILE_GET_FAPL:
            args->args.get_fapl.fapl_id = H5Pcopy(file->fapl_id);
            return args->args.get_fapl.fapl_id < 0 ? -1 : 0;
        case H5VL_FILE_GET_FCPL:
            args->args.get_fcpl.fcpl_id = H5Pcopy(file->fcpl_id);
            return args->args.get_fcpl.fcpl_id < 0 ? -1 : 0;
        case H5VL_FILE_GET_FILENO:
            *args->args.get_fileno.fileno = file->meta_id;
            return 0;
        case H5VL_FILE_GET_INTENT:
            *args->args.get_intent.flags = (file->flags & (H5F_ACC_RDWR | H5F_ACC_TRUNC | H5F_ACC_EXCL | H5F_ACC_CREAT))
                                               ? H5F_ACC_RDWR
                                               : H5F_ACC_RDONLY;
            return 0;
        case H5VL_FILE_GET_NAME: {
            size_t len = strlen(file->file_name);
            if (args->args.get_name.buf && args->args.get_name.buf_size > 0) {
                strncpy(args->args.get_name.buf, file->file_name, args->args.get_name.buf_size - 1);
                args->args.get_name.buf[args->args.get_name.buf_size - 1] = '\0';
            }
            *args->args.get_name.file_name_len = len;
            return 0;
        }
        case H5VL_FILE_GET_OBJ_COUNT:
            *args->args.get_obj_count.count = _obj_count(file, args->args.get_obj_count.types, 0, NULL);
            return 0;
        case H5VL_FILE_GET_OBJ_IDS:
            *args->args.get_obj_ids.count = _obj_count(file, args->args.get_obj_ids.types,
                                                       args->args.get_obj_ids.max_objs, args->args.get_obj_ids.oid_list);
            return 0;
        case H5VL_FILE_GET_CONT_INFO:
        default:
            return -1;
    }
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
    H5VL_pdc_obj_t *file = _file_of(o);
    int             ndim;
    H5T_class_t     dclass;
    H5VL_pdc_obj_t *dset = NULL;
    pdcid_t         obj_prop, obj_id, parent;
    hsize_t         dims[H5S_MAX_RANK];
    char *          parent_name = NULL, *leaf = NULL, *new_name = NULL;
    _pdc_loc_t      parent_loc;
    int             have_parent = 0;
    pdc_entity_type_t obj_type  = PDC_ENTITY_OBJ;

    FUNC_ENTER_VOL(void *, NULL)

#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entered dataset_create [%s][%s][%s]\n", o->my_rank, o->file_name, o->path, name);
#endif

    if (!name)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "dataset name is NULL");
    _split_name(name, &parent_name, &leaf);
    if (_resolve(o, parent_name, &parent_loc) != 0 || parent_loc.type != PDC_ENTITY_CONT)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, NULL, "parent group does not exist");
    have_parent = 1;
    file        = parent_loc.file;
    _change_begin(file);
    {
        herr_t status = 0;
        if (_is_writer(file) && _find_child((pdcid_t)parent_loc.meta_id, leaf, NULL, NULL, NULL, NULL) != 1)
            status = -1;
        if (_sync_status(file, status) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, NULL, "name already exists");
    }
    parent   = parent_loc.meta_id == o->meta_id ? o->cont_id : PDCcont_open_id((pdcid_t)parent_loc.meta_id, pdc_id_g);
    new_name = _pdc_obj_name(file, parent_loc.path, leaf);

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
    if (_is_coll(file)) {
#ifdef ENABLE_LOGGING
        fprintf(stderr, "Rank %d: PDC obj create mpi [%s]\n", o->my_rank, new_name);
#endif
        obj_id = PDCobj_create_coll(parent, new_name, obj_prop, 0, file->comm);
    }
    else {
#ifdef ENABLE_LOGGING
        fprintf(stderr, "Rank %d: PDC obj create [%s]\n", o->my_rank, new_name);
#endif
        obj_id = PDCobj_create(parent, new_name, obj_prop);
    }
    if (obj_id <= 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, NULL, "can't create PDC object");

    /* The parent lists its datasets; that listing is the link */
    if ((_is_coll(file) ? PDCcont_put_entries_coll(parent, 1, &obj_id, &obj_type, file->comm)
                        : PDCcont_put_entries(parent, 1, &obj_id, &obj_type)) != SUCCEED)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, NULL, "can't link PDC object into its parent");
    if (parent != o->cont_id)
        PDCcont_close(parent);

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
    dset->meta_id  = PDCobj_get_info(obj_id)->meta_id;
    dset->path     = _path_join(parent_loc.path, leaf);
    dset->pdc_name = strdup(new_name);
    dset->under_vol_id = file->under_vol_id;
    dset->under_object = dset;
    strncpy(dset->obj_name, leaf, sizeof(dset->obj_name) - 1);
    o->nobj++;

    if ((PDCprop_close(obj_prop)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, NULL, "can't close object property");

    /* Set return value */
    FUNC_RETURN_SET((void *)dset);

done:
    if (have_parent)
        _loc_free(&parent_loc);
    free(parent_name);
    free(leaf);
    free(new_name);
    FUNC_LEAVE_VOL
}

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_dataset_open(void *obj, const H5VL_loc_params_t *loc_params __attribute__((unused)), const char *_name,
                      hid_t dapl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                      void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o    = (H5VL_pdc_obj_t *)obj;
    H5VL_pdc_obj_t *dset = NULL;
    _pdc_loc_t      loc;

    if (_resolve(o, _name, &loc) != 0)
        return NULL;
    if (loc.type == PDC_ENTITY_OBJ)
        dset = _open_dset_obj(loc.file, &loc);
    _loc_free(&loc);
    return dset;
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
/* What a dataset transfer touches in the file: one box of the dataset. PDC regions
 * are boxes, so other selections are refused rather than written to the wrong place. */
static herr_t
_file_box(H5VL_pdc_obj_t *dset, hid_t file_space, int *ndim, uint64_t *off, uint64_t *cnt, hsize_t *npoints)
{
    hid_t   space = (file_space == H5S_ALL) ? dset->space_id : file_space;
    hsize_t dims[H5S_MAX_RANK], start[H5S_MAX_RANK], end[H5S_MAX_RANK], box = 1;
    H5S_sel_type sel = file_space == H5S_ALL ? H5S_SEL_ALL : H5Sget_select_type(space);

    if ((*ndim = H5Sget_simple_extent_dims(dset->space_id, dims, NULL)) < 0)
        return -1;
    if (*ndim > 4)
        return -1; /* PDC regions */
    if (sel == H5S_SEL_NONE) {
        *npoints = 0;
        return 0;
    }
    if (sel == H5S_SEL_ALL) {
        for (int i = 0; i < *ndim; i++) {
            off[i] = 0;
            cnt[i] = dims[i];
            box *= dims[i];
        }
    }
    else {
        hssize_t n = H5Sget_select_npoints(space);
        if (n < 0 || H5Sget_select_bounds(space, start, end) < 0)
            return -1;
        for (int i = 0; i < *ndim; i++) {
            off[i] = start[i];
            cnt[i] = end[i] - start[i] + 1;
            box *= cnt[i];
        }
        if ((hsize_t)n != box)
            return -1; /* not a single box */
    }
    *npoints = box;
    return 0;
}

/* Whether the memory selection is one contiguous run from the start of the buffer */
static int
_mem_is_contig(hid_t mem_space, hsize_t npoints)
{
    if (mem_space == H5S_ALL)
        return 1;
    if (H5Sget_select_type(mem_space) == H5S_SEL_ALL)
        return 1;
    return (hsize_t)H5Sget_simple_extent_npoints(mem_space) == npoints;
}

/* Datasets that no datatype conversion is needed for */
static int
_same_type(hid_t a, hid_t b)
{
    return H5Tequal(a, b) > 0 || H5Tget_class(a) == H5T_COMPOUND || H5Tget_class(b) == H5T_COMPOUND;
}

/* Convert n elements in buf (sized for the larger type). A string and a one-byte
 * integer hold the same bytes, and HDF5 will not convert between those classes. */
static herr_t
_convert(hid_t from, hid_t to, size_t n, void *buf)
{
    if (H5Tget_size(from) == H5Tget_size(to) &&
        (H5Tget_class(from) == H5T_STRING || H5Tget_class(to) == H5T_STRING))
        return 0;
    return H5Tconvert(from, to, n, buf, NULL, H5P_DEFAULT);
}

static herr_t
H5VL__pdc_gather_cb(const void *dst_buf __attribute__((unused)), size_t dst_buf_bytes_used __attribute__((unused)),
                    void *op_data __attribute__((unused)))
{
    return 0; /* called when the buffer fills; it is sized for the whole selection */
}

/* H5Dscatter asks a callback for the packed data; here it is one buffer, passed as op_data */
typedef struct _scatter_src_t {
    const void *buf;
    size_t      size;
} _scatter_src_t;

static herr_t
H5VL__pdc_scatter_src(const void **src_buf, size_t *src_buf_bytes_used, void *op_data)
{
    _scatter_src_t *src = (_scatter_src_t *)op_data;
    *src_buf            = src->buf;
    *src_buf_bytes_used = src->size;
    return 0;
}

herr_t
H5VL_pdc_dataset_write(size_t count, void *_dset[], hid_t mem_type_id[], hid_t mem_space_id[],
                       hid_t file_space_id[], hid_t plist_id __attribute__((unused)), const void *buf[],
                       void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif

    H5VL_pdc_obj_t *dset, *file;

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    for (size_t u = 0; u < count; u++) {
        uint64_t off[H5S_MAX_RANK] = {0}, cnt[H5S_MAX_RANK] = {0}, zero[H5S_MAX_RANK] = {0};
        hsize_t  npoints;
        int      ndim;
        size_t   mem_size, file_size, total;
        void *   cache_buf;
        pdcid_t  region_local, region_remote, transfer_request;

        dset = (H5VL_pdc_obj_t *)_dset[u];
        file = dset->file_obj_ptr;

        if (_file_box(dset, file_space_id[u], &ndim, off, cnt, &npoints) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "vol-pdc transfers a single box of a dataset");
        if (npoints == 0)
            continue;

        mem_size  = H5Tget_size(mem_type_id[u]);
        file_size = H5Tget_size(dset->type_id);
        total     = (size_t)npoints * (mem_size > file_size ? mem_size : file_size);
        cache_buf = malloc(total);

        if (_mem_is_contig(mem_space_id[u], npoints))
            memcpy(cache_buf, buf[u], (size_t)npoints * mem_size);
        else if (H5Dgather(mem_space_id[u], buf[u], mem_type_id[u], total, cache_buf, H5VL__pdc_gather_cb, NULL) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "can't gather memory selection");

        if (!_same_type(mem_type_id[u], dset->type_id) &&
            _convert(mem_type_id[u], dset->type_id, (size_t)npoints, cache_buf) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "can't convert datatype");

        // TODO: temporary workaround for writing compound data, as current PDC doesn't support
        //       compound datatype
        if (H5Tget_class(dset->type_id) == H5T_COMPOUND) {
            off[ndim - 1] *= file_size;
            cnt[ndim - 1] *= file_size;
        }

        region_local      = PDCregion_create(ndim, zero, cnt);
        dset->reg_id_from = region_local;
        region_remote     = PDCregion_create(ndim, off, cnt);
        dset->reg_id_to   = region_remote;

        transfer_request =
            PDCregion_transfer_create(cache_buf, PDC_WRITE, dset->obj_id, region_local, region_remote);
        if (transfer_request == 0)
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "can't create transfer");
        _add_xfer_request(file, transfer_request, cache_buf);
        write_cache_size_g += total;

        /* Deferred to the next read, flush or close, unless the cache is full */
        if (write_cache_size_g > MAX_WRITE_CACHE_SIZE_GB * 1073741824llu &&
            _flush_xfer_requests(file) != SUCCEED)
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "Failed to complete deferred writes");
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

    FUNC_ENTER_VOL(herr_t, SUCCEED)

    // Complete existing write requests
    dset = (H5VL_pdc_obj_t *)_dset[0];
    file = dset->file_obj_ptr;
    if (_flush_xfer_requests(file) != SUCCEED)
        HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "Failed to complete deferred writes");

    for (size_t u = 0; u < count; u++) {
        uint64_t off[H5S_MAX_RANK] = {0}, cnt[H5S_MAX_RANK] = {0}, zero[H5S_MAX_RANK] = {0};
        hsize_t  npoints;
        int      ndim, contig, direct;
        size_t   mem_size, file_size, total;
        void *   target;
        pdcid_t  region_local, region_remote, transfer_request;
        perr_t   ret;

        dset = (H5VL_pdc_obj_t *)_dset[u];

        if (_file_box(dset, file_space_id[u], &ndim, off, cnt, &npoints) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "vol-pdc transfers a single box of a dataset");
        if (npoints == 0)
            continue;

        mem_size  = H5Tget_size(mem_type_id[u]);
        file_size = H5Tget_size(dset->type_id);
        total     = (size_t)npoints * (mem_size > file_size ? mem_size : file_size);
        contig    = _mem_is_contig(mem_space_id[u], npoints);
        direct    = contig && _same_type(mem_type_id[u], dset->type_id);
        target    = direct ? buf[u] : malloc(total);

        // TODO: temporary workaround for reading compound data, as current PDC doesn't support
        //       compound datatype
        if (dset->compound_size > 0) {
            off[ndim - 1] *= dset->compound_size;
            cnt[ndim - 1] *= dset->compound_size;
        }

        region_local      = PDCregion_create(ndim, zero, cnt);
        dset->reg_id_from = region_local;
        region_remote     = PDCregion_create(ndim, off, cnt);
        dset->reg_id_to   = region_remote;

        transfer_request = PDCregion_transfer_create(target, PDC_READ, dset->obj_id, region_local, region_remote);
        if (transfer_request == 0)
            HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "can't create transfer");
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

        if (!direct) {
            if (!_same_type(mem_type_id[u], dset->type_id) &&
                _convert(dset->type_id, mem_type_id[u], (size_t)npoints, target) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "can't convert datatype");
            if (contig)
                memcpy(buf[u], target, (size_t)npoints * mem_size);
            else {
                _scatter_src_t src = {target, (size_t)npoints * mem_size};
                if (H5Dscatter(H5VL__pdc_scatter_src, &src, mem_type_id[u], mem_space_id[u], buf[u]) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "can't scatter to memory selection");
            }
            free(target);
        }
    }

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
H5VL_pdc_dataset_specific(void *obj, H5VL_dataset_specific_args_t *args, hid_t dxpl_id __attribute__((unused)),
                          void **req __attribute__((unused)))
{
    switch (args->op_type) {
        case H5VL_DATASET_FLUSH:
            return _flush_xfer_requests(_file_of((H5VL_pdc_obj_t *)obj)) == SUCCEED ? 0 : -1;
        case H5VL_DATASET_REFRESH:
            return 0;
        case H5VL_DATASET_SET_EXTENT:
        default:
            return -1;
    }
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

    {
        H5VL_pdc_obj_t *file = dset->file_obj_ptr;
        if (H5VL__pdc_dset_free(dset) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't free dataset");
        if (H5VL__pdc_file_close(file) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "can't release file");
    }

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
static herr_t
H5VL_pdc_introspect_get_conn_cls(void *obj __attribute__((unused)), H5VL_get_conn_lvl_t lvl __attribute__((unused)),
                                 const H5VL_class_t **conn_cls)
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    /* This is a terminal connector: there is nothing under it to ask */
    *conn_cls = &H5VL_pdc_g;
    return 0;
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
H5VL_pdc_group_create(void *obj, const H5VL_loc_params_t *loc_params __attribute__((unused)),
                      const char *name, hid_t lcpl_id __attribute__((unused)),
                      hid_t gcpl_id __attribute__((unused)), hid_t gapl_id __attribute__((unused)),
                      hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o     = (H5VL_pdc_obj_t *)obj;
    H5VL_pdc_obj_t *file  = _file_of(o);
    H5VL_pdc_obj_t *group = NULL;
    char *          parent_name, *leaf, *path = NULL;
    _pdc_loc_t      parent_loc;
    pdcid_t         parent = 0, prop = 0, child = 0;
    uint64_t        child_id;

    _split_name(name, &parent_name, &leaf);
    if (_resolve(o, parent_name, &parent_loc) != 0) {
        free(parent_name);
        free(leaf);
        return NULL;
    }
    if (parent_loc.type != PDC_ENTITY_CONT)
        goto done;
    file = parent_loc.file;
    _change_begin(file);
    {
        herr_t status = 0;
        if (_is_writer(file) && _find_child((pdcid_t)parent_loc.meta_id, leaf, NULL, NULL, NULL, NULL) != 1)
            status = -1;
        if (_sync_status(file, status) < 0)
            goto done;
    }

    parent = parent_loc.meta_id == o->meta_id && o->h5i_type != H5I_DATASET
                 ? o->cont_id
                 : PDCcont_open_id((pdcid_t)parent_loc.meta_id, pdc_id_g);
    prop   = PDCprop_create(PDC_CONT_CREATE, pdc_id_g);
    child  = _is_coll(file) ? PDCcont_create_at_coll(parent, leaf, prop, file->comm)
                            : PDCcont_create_at(parent, leaf, prop);
    if (child == 0 || _find_child(parent, leaf, &child_id, NULL, NULL, NULL) != 0)
        goto done;
    path  = _path_join(parent_loc.path, leaf);
    group = _open_group_obj(file, child_id, path, child);

done:
    if (prop)
        PDCprop_close(prop);
    if (parent && parent != o->cont_id)
        PDCcont_close(parent);
    _loc_free(&parent_loc);
    free(parent_name);
    free(leaf);
    free(path);
    return (void *)group;
} /* end H5VL_pdc_group_create() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_group_open(void *obj, const H5VL_loc_params_t *loc_params __attribute__((unused)), const char *name,
                    hid_t gapl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                    void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o     = (H5VL_pdc_obj_t *)obj;
    H5VL_pdc_obj_t *group = NULL;
    _pdc_loc_t      loc;

    if (_resolve(o, name, &loc) != 0)
        return NULL;
    if (loc.type == PDC_ENTITY_CONT)
        group = _open_group_obj(loc.file, loc.meta_id, loc.path, 0);
    _loc_free(&loc);
    return (void *)group;
} /* end H5VL_pdc_group_open() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_group_get(void *obj, H5VL_group_get_args_t *args, hid_t dxpl_id __attribute__((unused)),
                   void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;

    switch (args->op_type) {
        case H5VL_GROUP_GET_GCPL: {
            /* The entries of a container are in creation order */
            hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE);
            H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
            args->args.get_gcpl.gcpl_id = gcpl;
            return gcpl < 0 ? -1 : 0;
        }
        case H5VL_GROUP_GET_INFO: {
            H5VL_pdc_obj_t *tmp, *owner = _loc_owner(o, &args->args.get_info.loc_params, &tmp);
            int             n = 0;
            pdc_cont_entry_t *entries = NULL;
            herr_t          ret = -1;
            if (owner && owner->h5i_type != H5I_DATASET &&
                PDCcont_get_entries(owner->cont_id, &n, &entries) == SUCCEED) {
                H5G_info_t *info   = args->args.get_info.ginfo;
                info->storage_type = H5G_STORAGE_TYPE_DENSE;
                info->nlinks       = (hsize_t)n;
                info->max_corder   = (int64_t)n;
                info->mounted      = FALSE;
                free(entries);
                ret = 0;
            }
            if (tmp)
                _close_obj(tmp);
            return ret;
        }
        default:
            return -1;
    }
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
H5VL_pdc_group_close(void *grp, hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *group = (H5VL_pdc_obj_t *)grp;
    H5VL_pdc_obj_t *file  = group->file_obj_ptr;
    herr_t          ret   = 0;

    if (PDCcont_close(group->cont_id) != SUCCEED)
        ret = -1;
    free(group->path);
    free(group->group_name);
    free(group->file_name);
    free(group);
    if (H5VL__pdc_file_close(file) < 0)
        ret = -1;
    return ret;
} /* end H5VL_pdc_group_close() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_link_create(H5VL_link_create_args_t *args, void *obj, const H5VL_loc_params_t *loc_params,
                     hid_t lcpl_id __attribute__((unused)), hid_t lapl_id __attribute__((unused)),
                     hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    H5VL_pdc_obj_t *o    = (H5VL_pdc_obj_t *)obj;
    H5VL_pdc_obj_t *file = _file_of(o);
    const char *    target, *target_file = "";
    char            kind;
    char *          parent_name, *leaf, *blob;
    size_t          blob_len;
    _pdc_loc_t      parent_loc;
    pdcid_t         parent, prop, child;
    herr_t          status = 0;

    if (loc_params->type != H5VL_OBJECT_BY_NAME)
        return -1;
    if (args->op_type == H5VL_LINK_CREATE_SOFT) {
        kind   = 'S';
        target = args->args.soft.target;
    }
    else if (args->op_type == H5VL_LINK_CREATE_UD && args->args.ud.type == H5L_TYPE_EXTERNAL) {
        /* version and flags byte, file, NUL, path, NUL */
        kind        = 'E';
        target_file = (const char *)args->args.ud.buf + 1;
        target      = target_file + strlen(target_file) + 1;
    }
    else {
        return -1; /* hard links would need a second name for a container */
    }

    _split_name(loc_params->loc_data.loc_by_name.name, &parent_name, &leaf);
    if (_resolve(o, parent_name, &parent_loc) != 0) {
        free(parent_name);
        free(leaf);
        return -1;
    }
    _change_begin(file);
    if (parent_loc.type != PDC_ENTITY_CONT ||
        (_is_writer(file) && _find_child((pdcid_t)parent_loc.meta_id, leaf, NULL, NULL, NULL, NULL) != 1))
        status = -1;
    if (_sync_status(file, status) < 0) {
        _loc_free(&parent_loc);
        free(parent_name);
        free(leaf);
        return -1;
    }

    parent = parent_loc.meta_id == o->meta_id && o->h5i_type != H5I_DATASET
                 ? o->cont_id
                 : PDCcont_open_id((pdcid_t)parent_loc.meta_id, pdc_id_g);
    prop   = PDCprop_create(PDC_CONT_CREATE, pdc_id_g);
    child  = _is_coll(file) ? PDCcont_create_at_coll(parent, leaf, prop, file->comm)
                            : PDCcont_create_at(parent, leaf, prop);
    status = child ? 0 : -1;
    if (child && _is_writer(file)) {
        blob_len = 1 + strlen(target) + 1 + strlen(target_file) + 1;
        blob     = (char *)malloc(blob_len);
        blob[0]  = kind;
        strcpy(blob + 1, target);
        strcpy(blob + 2 + strlen(target), target_file);
        status = PDCcont_put_tag(child, H5VL_PDC_LINK_TAG, blob, PDC_CHAR, blob_len) == SUCCEED ? 0 : -1;
        free(blob);
    }
    status = _sync_status(file, status);
    if (child)
        PDCcont_close(child);
    PDCprop_close(prop);
    if (parent != o->cont_id)
        PDCcont_close(parent);
    _loc_free(&parent_loc);
    free(parent_name);
    free(leaf);
    return status;
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
    return -1;
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
    return -1;
} /* end H5VL_pdc_link_move() */

/*---------------------------------------------------------------------------*/
/* Type, and token or value size, of a link */
static void
_fill_link_info(uint64_t id, H5L_info2_t *linfo)
{
    H5L_type_t  type;
    const char *path, *file;

    if (_entry_link(id, &type, &path, &file)) {
        linfo->type = type;
        linfo->u.val_size =
            type == H5L_TYPE_SOFT ? strlen(path) + 1 : 1 + strlen(file) + 1 + strlen(path) + 1;
    }
    else {
        linfo->type = H5L_TYPE_HARD;
        _token_of(id, &linfo->u.token);
    }
}

static herr_t
H5VL_pdc_link_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_get_args_t *args,
                  hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;

    if (args->op_type == H5VL_LINK_GET_INFO && loc_params->type == H5VL_OBJECT_BY_NAME) {
        char *       parent_name, *leaf;
        _pdc_loc_t   parent_loc;
        uint64_t     id;
        hsize_t      position;
        herr_t       ret = -1;
        H5L_info2_t *linfo = args->args.get_info.linfo;

        _split_name(loc_params->loc_data.loc_by_name.name, &parent_name, &leaf);
        if (_resolve(o, parent_name, &parent_loc) == 0) {
            if (parent_loc.type == PDC_ENTITY_CONT &&
                _find_child((pdcid_t)parent_loc.meta_id, leaf, &id, NULL, NULL, &position) == 0) {
                memset(linfo, 0, sizeof(*linfo));
                linfo->corder_valid = TRUE;
                linfo->corder       = (int64_t)position;
                linfo->cset         = H5T_CSET_ASCII;
                _fill_link_info(id, linfo);
                ret = 0;
            }
            _loc_free(&parent_loc);
        }
        free(parent_name);
        free(leaf);
        return ret;
    }
    if (args->op_type == H5VL_LINK_GET_VAL && loc_params->type == H5VL_OBJECT_BY_NAME) {
        char *      parent_name, *leaf;
        _pdc_loc_t  parent_loc;
        uint64_t    id;
        herr_t      ret = -1;
        H5L_type_t  type;
        const char *path, *file;

        _split_name(loc_params->loc_data.loc_by_name.name, &parent_name, &leaf);
        if (_resolve(o, parent_name, &parent_loc) == 0) {
            if (parent_loc.type == PDC_ENTITY_CONT &&
                _find_child((pdcid_t)parent_loc.meta_id, leaf, &id, NULL, NULL, NULL) == 0 &&
                _entry_link(id, &type, &path, &file)) {
                size_t size = type == H5L_TYPE_SOFT ? strlen(path) + 1
                                                    : 1 + strlen(file) + 1 + strlen(path) + 1;
                char * val  = (char *)malloc(size);
                if (type == H5L_TYPE_SOFT)
                    strcpy(val, path);
                else {
                    /* the format H5Lunpack_elink_val reads */
                    val[0] = (char)((H5L_EXT_VERSION << 4) | 0);
                    strcpy(val + 1, file);
                    strcpy(val + 2 + strlen(file), path);
                }
                if (args->args.get_val.buf && args->args.get_val.buf_size > 0)
                    memcpy(args->args.get_val.buf, val,
                           size < args->args.get_val.buf_size ? size : args->args.get_val.buf_size);
                free(val);
                ret = 0;
            }
            _loc_free(&parent_loc);
        }
        free(parent_name);
        free(leaf);
        return ret;
    }
    return -1;
} /* end H5VL_pdc_link_get() */

/*---------------------------------------------------------------------------*/
typedef struct _link_rec_t {
    char *   name;
    uint64_t meta_id;
    hsize_t  corder;
} _link_rec_t;

static int
_link_rec_cmp(const void *a, const void *b)
{
    return strcmp(((const _link_rec_t *)a)->name, ((const _link_rec_t *)b)->name);
}

static herr_t
_link_iterate(H5VL_pdc_obj_t *owner, H5VL_link_iterate_args_t *it)
{
    H5VL_pdc_obj_t *  file = _file_of(owner);
    int               n = 0, nrec = 0;
    pdc_cont_entry_t *entries = NULL;
    _link_rec_t *     recs;
    herr_t            ret = 0;
    hsize_t           start;
    H5VL_pdc_obj_t *  self;
    hid_t             gid;

    if (it->recursive || owner->h5i_type == H5I_DATASET)
        return -1;
    if (PDCcont_get_entries(owner->cont_id, &n, &entries) != SUCCEED)
        return -1;
    recs = (_link_rec_t *)calloc(n ? n : 1, sizeof(_link_rec_t));
    for (int i = 0; i < n; i++) {
        const char *name = _entry_name(entries[i].meta_id);
        if (name) {
            recs[nrec].name    = _leaf_of(name);
            recs[nrec].meta_id = entries[i].meta_id;
            recs[nrec].corder  = (hsize_t)nrec;
            nrec++;
        }
    }
    free(entries);
    if (it->idx_type == H5_INDEX_NAME)
        qsort(recs, nrec, sizeof(_link_rec_t), _link_rec_cmp);

    /* The callback takes an ID for the group being iterated */
    self = _open_group_obj(file, owner->meta_id, owner->path ? owner->path : "/", 0);
    gid  = self ? H5VLwrap_register(self, H5I_GROUP) : H5I_INVALID_HID;
    if (gid < 0)
        ret = -1;

    start = it->idx_p ? *it->idx_p : 0;
    for (hsize_t k = start; ret == 0 && k < (hsize_t)nrec; k++) {
        _link_rec_t *r = &recs[it->order == H5_ITER_DEC ? (hsize_t)nrec - 1 - k : k];
        H5L_info2_t  linfo;

        memset(&linfo, 0, sizeof(linfo));
        linfo.corder_valid = TRUE;
        linfo.corder       = (int64_t)r->corder;
        linfo.cset         = H5T_CSET_ASCII;
        _fill_link_info(r->meta_id, &linfo);
        ret = it->op(gid, r->name, &linfo, it->op_data);
        if (it->idx_p)
            *it->idx_p = k + 1;
    }
    if (gid >= 0)
        H5Idec_ref(gid);
    for (int i = 0; i < nrec; i++)
        free(recs[i].name);
    free(recs);
    return ret;
}

static herr_t
H5VL_pdc_link_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_specific_args_t *args,
                       hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    H5VL_pdc_obj_t *o    = (H5VL_pdc_obj_t *)obj;
    H5VL_pdc_obj_t *file = _file_of(o);

    switch (args->op_type) {
        case H5VL_LINK_EXISTS: {
            _pdc_loc_t loc;
            int        found;
            if (loc_params->type != H5VL_OBJECT_BY_NAME)
                return -1;
            found = _resolve_o(o, loc_params->loc_data.loc_by_name.name, 0, &loc);
            if (found < 0)
                return -1;
            if (found == 0)
                _loc_free(&loc);
            *args->args.exists.exists = found == 0;
            return 0;
        }
        case H5VL_LINK_ITER: {
            H5VL_pdc_obj_t *tmp, *owner = _loc_owner(o, loc_params, &tmp);
            herr_t          ret;
            if (owner == NULL)
                return -1;
            ret = _link_iterate(owner, &args->args.iterate);
            if (tmp)
                _close_obj(tmp);
            return ret;
        }
        case H5VL_LINK_DELETE: {
            char *            parent_name, *leaf, *pdc_name = NULL;
            _pdc_loc_t        parent_loc;
            uint64_t          id;
            pdc_entity_type_t type;
            herr_t            status = -1;

            if (loc_params->type != H5VL_OBJECT_BY_NAME)
                return -1;
            if (_flush_xfer_requests(file) != SUCCEED)
                return -1;
            _change_begin(file);
            _split_name(loc_params->loc_data.loc_by_name.name, &parent_name, &leaf);
            if (_is_writer(file) && _resolve(o, parent_name, &parent_loc) == 0) {
                if (parent_loc.type == PDC_ENTITY_CONT &&
                    _find_child((pdcid_t)parent_loc.meta_id, leaf, &id, &type, &pdc_name, NULL) == 0) {
                    pdcid_t parent = PDCcont_open_id((pdcid_t)parent_loc.meta_id, pdc_id_g);
                    if (parent) {
                        status = _delete_entry(parent, id, type, pdc_name);
                        PDCcont_close(parent);
                    }
                    free(pdc_name);
                }
                _loc_free(&parent_loc);
            }
            free(parent_name);
            free(leaf);
            return _sync_status(file, status);
        }
        default:
            return -1;
    }
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
static H5VL_pdc_obj_t *
_new_attr_obj(H5VL_pdc_obj_t *owner, H5VL_pdc_obj_t *tmp, const char *name, hid_t type_id, hid_t space_id)
{
    H5VL_pdc_obj_t *file = _file_of(owner);
    H5VL_pdc_obj_t *attr = H5VL_pdc_new_obj(NULL, owner->under_vol_id);

    attr->under_object    = attr;
    attr->attr_name       = strdup(name);
    attr->attr_type       = type_id;
    attr->attr_space      = space_id;
    attr->attr_value_size = (psize_t)H5Sget_simple_extent_npoints(space_id) * H5Tget_size(type_id);
    attr->obj_id          = owner->obj_id;
    attr->cont_id         = owner->cont_id;
    attr->file_obj_ptr    = file;
    attr->h5i_type        = H5I_ATTR;
    attr->attr_owner_tmp  = tmp;
    file->refcnt++;
    return attr;
}

static void *
H5VL_pdc_attr_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t type_id,
                     hid_t space_id, hid_t acpl_id __attribute__((unused)),
                     hid_t aapl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                     void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj, *tmp;
    H5VL_pdc_obj_t *owner = _loc_owner(o, loc_params, &tmp);
    H5VL_pdc_obj_t *file  = _file_of(o);
    herr_t          status = -1;

    if (owner == NULL)
        return NULL;
    _change_begin(file);
    if (_is_writer(file)) {
        void *  idx = NULL;
        psize_t len;
        if (_tag_get(owner, H5VL_PDC_ATTR_INDEX, &idx, &len, NULL) == 0 &&
            !(idx && _attr_find(idx, len, name, NULL, NULL)))
            status = _attr_index_add(owner, name, type_id, space_id);
        free(idx);
    }
    if (_sync_status(file, status) < 0) {
        if (tmp)
            _close_obj(tmp);
        return NULL;
    }
    return (void *)_new_attr_obj(owner, tmp, name, H5Tcopy(type_id), H5Scopy(space_id));
} /* end H5VL_pdc_attr_create() */
/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_attr_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name,
                   hid_t aapl_id __attribute__((unused)), hid_t dxpl_id __attribute__((unused)),
                   void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj, *tmp, *attr = NULL;
    H5VL_pdc_obj_t *owner = _loc_owner(o, loc_params, &tmp);
    void *          idx   = NULL;
    psize_t         len;
    _attr_rec_t     rec;

    if (owner == NULL)
        return NULL;
    if (_tag_get(owner, H5VL_PDC_ATTR_INDEX, &idx, &len, NULL) == 0 && idx && _attr_find(idx, len, name, &rec, NULL))
        attr = _new_attr_obj(owner, tmp, name, H5Tdecode(rec.type_buf), H5Sdecode(rec.space_buf));
    else if (tmp)
        _close_obj(tmp);
    free(idx);
    return (void *)attr;
} /* end H5VL_pdc_attr_open() */
/*---------------------------------------------------------------------------*/
static perr_t
H5VL_pdc_attr_read(void *attr, hid_t mem_type_id, void *buf, hid_t dxpl_id __attribute__((unused)),
                   void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o         = (H5VL_pdc_obj_t *)attr;
    void *          tag_value = NULL;
    psize_t         size      = 0;
    size_t          npoints   = (size_t)H5Sget_simple_extent_npoints(o->attr_space);
    size_t          file_size = H5Tget_size(o->attr_type), mem_size = H5Tget_size(mem_type_id);
    herr_t          ret = 0;

    if (_tag_get(o, o->attr_name, &tag_value, &size, NULL) < 0)
        return -1;
    if (tag_value == NULL) {
        /* created but never written: HDF5 returns the fill value */
        memset(buf, 0, npoints * mem_size);
        return 0;
    }
    if (H5Tequal(mem_type_id, o->attr_type) > 0 || H5Tget_class(mem_type_id) == H5T_STRING ||
        H5Tget_class(o->attr_type) == H5T_STRING) {
        /* no numeric conversion; a string is bytes up to the smaller size */
        size_t want = npoints * mem_size;
        memset(buf, 0, want);
        memcpy(buf, tag_value, want < (size_t)size ? want : (size_t)size);
    }
    else {
        size_t         tconv = npoints * (file_size > mem_size ? file_size : mem_size);
        unsigned char *conv  = (unsigned char *)malloc(tconv);
        memcpy(conv, tag_value, npoints * file_size);
        ret = H5Tconvert(o->attr_type, mem_type_id, npoints, conv, NULL, H5P_DEFAULT);
        if (ret >= 0)
            memcpy(buf, conv, npoints * mem_size);
        free(conv);
    }
    free(tag_value);
    return ret < 0 ? -1 : 0;
} /* end H5VL_pdc_attr_read() */

/*---------------------------------------------------------------------------*/
static perr_t
H5VL_pdc_attr_write(void *attr, hid_t mem_type_id, const void *buf, hid_t dxpl_id __attribute__((unused)),
                    void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o         = (H5VL_pdc_obj_t *)attr;
    H5VL_pdc_obj_t *file      = o->file_obj_ptr;
    size_t          npoints   = (size_t)H5Sget_simple_extent_npoints(o->attr_space);
    size_t          file_size = H5Tget_size(o->attr_type), mem_size = H5Tget_size(mem_type_id);
    herr_t          status    = 0;

    _change_begin(file);
    if (_is_writer(file)) {
        if (H5Tequal(mem_type_id, o->attr_type) > 0 || H5Tget_class(mem_type_id) == H5T_STRING ||
            H5Tget_class(o->attr_type) == H5T_STRING) {
            size_t         stored = npoints * file_size, have = npoints * mem_size;
            unsigned char *value  = (unsigned char *)calloc(stored ? stored : 1, 1);
            memcpy(value, buf, have < stored ? have : stored);
            status = _tag_put(o, o->attr_name, value, stored);
            free(value);
        }
        else {
            size_t         tconv = npoints * (file_size > mem_size ? file_size : mem_size);
            unsigned char *conv  = (unsigned char *)malloc(tconv);
            memcpy(conv, buf, npoints * mem_size);
            status = H5Tconvert(mem_type_id, o->attr_type, npoints, conv, NULL, H5P_DEFAULT);
            if (status >= 0)
                status = _tag_put(o, o->attr_name, conv, npoints * file_size);
            free(conv);
        }
    }
    return _sync_status(file, status);
} /* end H5VL_pdc_attr_write() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_attr_get(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id __attribute__((unused)),
                  void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;

    switch (args->op_type) {
        case H5VL_ATTR_GET_SPACE:
            args->args.get_space.space_id = H5Scopy(o->attr_space);
            return args->args.get_space.space_id < 0 ? -1 : 0;
        case H5VL_ATTR_GET_TYPE:
            args->args.get_type.type_id = H5Tcopy(o->attr_type);
            return args->args.get_type.type_id < 0 ? -1 : 0;
        case H5VL_ATTR_GET_ACPL:
            args->args.get_acpl.acpl_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
            return args->args.get_acpl.acpl_id < 0 ? -1 : 0;
        case H5VL_ATTR_GET_NAME: {
            size_t len;
            if (args->args.get_name.loc_params.type != H5VL_OBJECT_BY_SELF)
                return -1;
            len = strlen(o->attr_name);
            if (args->args.get_name.buf && args->args.get_name.buf_size > 0) {
                strncpy(args->args.get_name.buf, o->attr_name, args->args.get_name.buf_size - 1);
                args->args.get_name.buf[args->args.get_name.buf_size - 1] = '\0';
            }
            *args->args.get_name.attr_name_len = len;
            return 0;
        }
        case H5VL_ATTR_GET_INFO: {
            H5A_info_t *info;
            if (args->args.get_info.loc_params.type != H5VL_OBJECT_BY_SELF)
                return -1;
            info = args->args.get_info.ainfo;
            memset(info, 0, sizeof(*info));
            info->cset      = H5T_CSET_ASCII;
            info->data_size = o->attr_value_size;
            return 0;
        }
        default:
            return -1;
    }
} /* end H5VL_pdc_attr_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_attr_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_attr_specific_args_t *args,
                       hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj, *tmp;
    H5VL_pdc_obj_t *owner = _loc_owner(o, loc_params, &tmp);
    H5VL_pdc_obj_t *file  = _file_of(o);
    void *          idx   = NULL;
    psize_t         len;
    herr_t          ret = -1;

    if (owner == NULL)
        return -1;

    switch (args->op_type) {
        case H5VL_ATTR_EXISTS:
            if (_tag_get(owner, H5VL_PDC_ATTR_INDEX, &idx, &len, NULL) == 0) {
                *args->args.exists.exists = idx && _attr_find(idx, len, args->args.exists.name, NULL, NULL);
                ret                       = 0;
            }
            break;
        case H5VL_ATTR_DELETE: {
            herr_t status = -1;
            _change_begin(file);
            if (_is_writer(file))
                status = _attr_index_remove(owner, args->args.del.name);
            ret = _sync_status(file, status);
            break;
        }
        case H5VL_ATTR_ITER: {
            H5VL_attr_iterate_args_t *it = &args->args.iterate;
            hid_t                     id = H5I_INVALID_HID;
            H5VL_pdc_obj_t *          self;
            _pdc_loc_t                loc = {0};
            int                       count = 0, total = 0, pos = 0;
            const unsigned char *     p, *end;
            _attr_rec_t               rec;
            hsize_t                   start = it->idx ? *it->idx : 0;

            /* The callback takes an ID for the attribute's owner */
            loc.type     = owner->h5i_type == H5I_DATASET ? PDC_ENTITY_OBJ : PDC_ENTITY_CONT;
            loc.meta_id  = owner->meta_id;
            loc.path     = owner->path;
            loc.pdc_name = owner->pdc_name;
            self         = _open_loc(file, &loc);
            if (self)
                id = H5VLwrap_register(self, owner->h5i_type == H5I_DATASET ? H5I_DATASET : H5I_GROUP);
            if (id < 0)
                break;
            ret = 0;
            if (_tag_get(owner, H5VL_PDC_ATTR_INDEX, &idx, &len, NULL) < 0)
                ret = -1;
            if (idx) {
                p   = idx;
                end = p + len;
                while ((p = _attr_next(p, end, &rec)) != NULL)
                    total++;
                (void)count;
                for (hsize_t k = start; ret == 0 && k < (hsize_t)total; k++) {
                    hsize_t    want = it->order == H5_ITER_DEC ? (hsize_t)total - 1 - k : k;
                    H5A_info_t ainfo;
                    char *     name;
                    p   = idx;
                    pos = 0;
                    while ((p = _attr_next(p, end, &rec)) != NULL && (hsize_t)pos != want)
                        pos++;
                    name = strndup(rec.name, rec.name_len);
                    memset(&ainfo, 0, sizeof(ainfo));
                    ainfo.cset = H5T_CSET_ASCII;
                    ret        = it->op(id, name, &ainfo, it->op_data);
                    free(name);
                    if (it->idx)
                        *it->idx = k + 1;
                }
            }
            H5Idec_ref(id);
            break;
        }
        default:
            break;
    }
    free(idx);
    if (tmp)
        _close_obj(tmp);
    return ret;
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
H5VL_pdc_attr_close(void *attr, hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o    = (H5VL_pdc_obj_t *)attr;
    H5VL_pdc_obj_t *file = o->file_obj_ptr;

    if (o->attr_type > 0)
        H5Tclose(o->attr_type);
    if (o->attr_space > 0)
        H5Sclose(o->attr_space);
    if (o->attr_owner_tmp)
        _close_obj(o->attr_owner_tmp);
    free(o->attr_name);
    free(o);
    return H5VL__pdc_file_close(file);
} /* end H5VL_pdc_attr_close() */

/*---------------------------------------------------------------------------*/
static void *
H5VL_pdc_object_open(void *obj, const H5VL_loc_params_t *loc_params, H5I_type_t *opened_type,
                     hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o       = (H5VL_pdc_obj_t *)obj;
    H5VL_pdc_obj_t *new_obj = NULL;
    _pdc_loc_t      loc;

    if (loc_params->type != H5VL_OBJECT_BY_NAME)
        return NULL;
    if (_resolve(o, loc_params->loc_data.loc_by_name.name, &loc) != 0)
        return NULL;
    new_obj = _open_loc(loc.file, &loc);
    if (new_obj)
        *opened_type = loc.type == PDC_ENTITY_CONT ? H5I_GROUP : H5I_DATASET;
    _loc_free(&loc);
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
H5VL_pdc_object_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_get_args_t *args,
                    hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj, *tmp = NULL, *target;
    herr_t          ret = -1;

    if (args->op_type == H5VL_OBJECT_GET_FILE) {
        *(args->args.get_file.file) = o->file_obj_ptr;
        return 0;
    }
    if (o->h5i_type == H5I_ATTR)
        return -1;
    target = _loc_owner(o, loc_params, &tmp);
    if (target == NULL)
        return -1;

    switch (args->op_type) {
        case H5VL_OBJECT_GET_NAME: {
            const char *path = target->path ? target->path : "/";
            size_t      len  = strlen(path);
            if (args->args.get_name.buf && args->args.get_name.buf_size > 0) {
                strncpy(args->args.get_name.buf, path, args->args.get_name.buf_size - 1);
                args->args.get_name.buf[args->args.get_name.buf_size - 1] = '\0';
            }
            *args->args.get_name.name_len = len;
            ret                           = 0;
            break;
        }
        case H5VL_OBJECT_GET_TYPE:
            *(args->args.get_type.obj_type) =
                target->h5i_type == H5I_DATASET ? H5O_TYPE_DATASET : H5O_TYPE_GROUP;
            ret = 0;
            break;
        case H5VL_OBJECT_GET_INFO: {
            hsize_t num_attrs = args->args.get_info.fields & H5O_INFO_NUM_ATTRS ? _attr_count(target) : 0;
            _fill_oinfo(_file_of(target), target->meta_id,
                        target->h5i_type == H5I_DATASET ? H5O_TYPE_DATASET : H5O_TYPE_GROUP, num_attrs,
                        args->args.get_info.oinfo);
            ret = 0;
            break;
        }
        default:
            break;
    }
    if (tmp)
        _close_obj(tmp);
    return ret;
} /* end H5VL_pdc_object_get() */

/*---------------------------------------------------------------------------*/
static herr_t
H5VL_pdc_object_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_specific_args_t *args,
                         hid_t dxpl_id __attribute__((unused)), void **req __attribute__((unused)))
{
#ifdef ENABLE_LOGGING
    fprintf(stderr, "Rank %d: entering %s\n", my_rank_g, __func__);
#endif
    H5VL_pdc_obj_t *o = (H5VL_pdc_obj_t *)obj;

    switch (args->op_type) {
        case H5VL_OBJECT_EXISTS: {
            _pdc_loc_t loc;
            int        found;
            if (loc_params->type != H5VL_OBJECT_BY_NAME)
                return -1;
            found = _resolve(o, loc_params->loc_data.loc_by_name.name, &loc);
            if (found < 0)
                return -1;
            if (found == 0)
                _loc_free(&loc);
            *args->args.exists.exists = found == 0;
            return 0;
        }
        case H5VL_OBJECT_FLUSH:
            return _flush_xfer_requests(_file_of(o)) == SUCCEED ? 0 : -1;
        case H5VL_OBJECT_REFRESH:
            return 0;
        case H5VL_OBJECT_CHANGE_REF_COUNT:
        case H5VL_OBJECT_LOOKUP:
        case H5VL_OBJECT_VISIT:
        default:
            return -1;
    }
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
H5VL_pdc_token_cmp(void *obj __attribute__((unused)), const H5O_token_t *token1, const H5O_token_t *token2,
                   int *cmp_value)
{
    *cmp_value = memcmp(token1, token2, sizeof(H5O_token_t));
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
