/*
 *  Copyright (c) 2014, Los Alamos National Security, LLC.
 *  All rights Reserved.
 *
 *  Copyright 2011-2012. Los Alamos National Security, LLC. This software was produced 
 *  under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National 
 *  Laboratory (LANL), which is operated by Los Alamos National Security, LLC 
 *  for the U.S. Department of Energy. The U.S. Government has rights to use, 
 *  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS 
 *  ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR 
 *  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified
 *  to produce derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Los Alamos National Security, LLC, Los Alamos 
 *       National Laboratory, LANL, the U.S. Government, nor the names of its 
 *       contributors may be used to endorse or promote products derived from 
 *       this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE LOS ALAMOS NATIONAL SECURITY, LLC AND 
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT 
 *  NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
 *  SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *  
 *  CLAMR -- LA-CC-11-094
 *  
 *  Authors: Brian Atkinson          bwa@g.clemson.edu
             Bob Robey        XCP-2  brobey@lanl.gov
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <assert.h>
#include "PowerParser/PowerParser.hh"

#include "crux.h"
#include "timer/timer.h"
#include "fmemopen.h"

#ifdef HAVE_HDF5
#include "hdf5.h"
#endif
#ifdef HAVE_MPI
#include "mpi.h"
#endif

const bool CRUX_TIMING = true;
bool do_crux_timing = false;

bool h5_spoutput;

#define RESTORE_NONE     0
#define RESTORE_RESTART  1
#define RESTORE_ROLLBACK 2

#ifndef DEBUG
#define DEBUG 0
#endif

using namespace std;
using PP::PowerParser;
// Pointers to the various objects.
PowerParser *parse;

char checkpoint_directory[] = "checkpoint_output";
FILE *store_fp, *restore_fp;
int cp_num, rs_num;
int *backup;
void **crux_data;
size_t *crux_data_size;
#ifdef HAVE_HDF5
bool USE_HDF5 = true; //MSB
hid_t h5_fid;
hid_t h5_gid_c, h5_gid_m, h5_gid_s;
herr_t h5err;
#endif
#ifdef HDF5_FF
hid_t e_stack;
hid_t tid1;
hid_t rid1, rid2;
#endif

FILE *crux_time_fp;
struct timeval tcheckpoint_time;
struct timeval trestore_time;
int checkpoint_timing_count = 0;
float checkpoint_timing_sum = 0.0f;
float checkpoint_timing_size = 0.0f;
int rollback_attempt = 0;

Crux::Crux(int crux_type_in, int num_of_rollback_states_in, bool restart)
{
   num_of_rollback_states = num_of_rollback_states_in;
   crux_type = crux_type_in;
   checkpoint_counter = 0;

   if (crux_type != CRUX_NONE || restart){
      do_crux_timing = CRUX_TIMING;
      struct stat stat_descriptor;
      if (stat(checkpoint_directory,&stat_descriptor) == -1){
        mkdir(checkpoint_directory,0777);
      }
   }

   crux_data = (void **)malloc(num_of_rollback_states*sizeof(void *));
   for (int i = 0; i < num_of_rollback_states; i++){
      crux_data[i] = NULL;
   }
   crux_data_size = (size_t *)malloc(num_of_rollback_states*sizeof(size_t));


   if (do_crux_timing){
      char checkpointtimelog[60];
      sprintf(checkpointtimelog,"%s/crux_timing.log",checkpoint_directory);
      crux_time_fp = fopen(checkpointtimelog,"w");
   }
}

Crux::~Crux()
{
   for (int i = 0; i < num_of_rollback_states; i++){
      free(crux_data[i]);
   }
   free(crux_data);
   free(crux_data_size);

   if (do_crux_timing){
      if (checkpoint_timing_count > 0) {
         printf("CRUX checkpointing time averaged %f msec, bandwidth %f Mbytes/sec\n",
                checkpoint_timing_sum/(float)checkpoint_timing_count*1.0e3,
                checkpoint_timing_size/checkpoint_timing_sum*1.0e-6);

         fprintf(crux_time_fp,"CRUX checkpointing time averaged %f msec, bandwidth %f Mbytes/sec\n",
                checkpoint_timing_sum/(float)checkpoint_timing_count*1.0e3,
                checkpoint_timing_size/checkpoint_timing_sum*1.0e-6);

      fclose(crux_time_fp);
      }
   }
}

void Crux::store_MallocPlus(MallocPlus memory){
   malloc_plus_memory_entry *memory_item;

   int mype = 0;
   int npes = 1;

#ifdef HAVE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD,&mype);
   MPI_Comm_size(MPI_COMM_WORLD,&npes);
#endif

#if HDF5_FF
  H5ES_status_t status;
  size_t num_events = 0;
  MPI_Request mpi_req;
  herr_t ret;
  uint64_t trans_num;
  hid_t mem_space_id;
#endif

   for (memory_item = memory.memory_entry_by_name_begin(); 
      memory_item != memory.memory_entry_by_name_end();
      memory_item = memory.memory_entry_by_name_next() ){

      void *mem_ptr = memory_item->mem_ptr;
      if ((memory_item->mem_flags & RESTART_DATA) == 0) continue;

      int num_elements = 1;
      for (uint i = 1; i < memory_item->mem_ndims; i++){
	 num_elements *= memory_item->mem_nelem[i];
      }

      if (DEBUG) {
        printf("MallocPlus ptr  %p: name %10s ptr %p dims %lu nelem (",
           mem_ptr,memory_item->mem_name,memory_item->mem_ptr,memory_item->mem_ndims);

        char nelemstring[80];
        char *str_ptr = nelemstring;
        str_ptr += sprintf(str_ptr,"%lu", memory_item->mem_nelem[0]);
        for (uint i = 1; i < memory_item->mem_ndims; i++){
           str_ptr += sprintf(str_ptr,", %lu", memory_item->mem_nelem[i]);
        }
        printf("%12s",nelemstring);

        printf(") elsize %lu flags %d capacity %lu\n",
           memory_item->mem_elsize,memory_item->mem_flags,memory_item->mem_capacity);
      }

#ifdef HAVE_HDF5
      if(USE_HDF5) {
        //
        // Create dataspace.  Setting maximum size to NULL sets the maximum
        // size to be the current size.
        //
        
        hid_t sid;
        sid = H5Screate_simple (memory_item->mem_ndims, (hsize_t*)memory_item->mem_nelem, NULL);
        
        hid_t memtype;
        hid_t filetype;
        if (memory_item->mem_elsize == 4){
          memtype = H5T_NATIVE_INT;
          filetype = H5T_NATIVE_INT;
        } else {
          memtype = H5T_NATIVE_DOUBLE;
          if(h5_spoutput) {
            filetype = H5T_NATIVE_FLOAT;
          } else {
            filetype = H5T_NATIVE_DOUBLE;
          }
        }
        
        hid_t gid;
        if( (strstr(memory_item->mem_name,"mesh") !=NULL) ||
            (strncmp(memory_item->mem_name,"i",1) == 0) || 
            (strncmp(memory_item->mem_name,"j",1) == 0) || 
            (strncmp(memory_item->mem_name,"level",5) == 0) ) {
          gid = h5_gid_m;
        }
        else if( (strstr(memory_item->mem_name,"state") != NULL) || 
                 (strncmp(memory_item->mem_name,"H",1) == 0) ||
                 (strncmp(memory_item->mem_name,"U",1) == 0) ||
                 (strncmp(memory_item->mem_name,"V",1) == 0) ) {
          gid = h5_gid_s; 
        }
        else {
          gid = h5_gid_c;
        }

        hid_t plist_id;
        plist_id = H5P_DEFAULT;
#ifdef HAVE_MPI
        //
        // Create property list for collective dataset write.
        //
        plist_id = H5Pcreate(H5P_DATASET_XFER);

#  ifdef HDF5_FF

// 	tid1 = H5TRcreate(h5_fid, rid2, (uint64_t)2);

// 	trans_num = 3;
// 	printf("H5TRstart 3\n");
// 	ret = H5TRstart(tid1, H5P_DEFAULT, H5_EVENT_STACK_NULL);
#  else
 	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#  endif

#endif
        if( (strncmp(memory_item->mem_name,"state_long_vals",15) == 0) ||
            (strncmp(memory_item->mem_name,"mesh_double_vals",16) == 0) ) {
          hid_t aid;
#ifdef HDF5_FF
	  if(mype == 0) {
	    aid = H5Acreate_ff(gid, memory_item->mem_name, filetype, sid, 
			       H5P_DEFAULT, H5P_DEFAULT, tid1, e_stack);
	    assert(aid > 0);
          // Write the attribute data.
	    h5err = H5Awrite_ff(aid, filetype, mem_ptr, tid1, e_stack);
	    h5err = H5Aclose_ff(aid, e_stack);
	  }
#else
	  aid = H5Acreate2(gid, memory_item->mem_name, filetype, sid, H5P_DEFAULT, H5P_DEFAULT);
          // Write the attribute data.
          h5err = H5Awrite(aid, filetype, mem_ptr);
          h5err = H5Aclose(aid);
#endif
        } else if( (strstr(memory_item->mem_name,"_timer") !=NULL) ||
                   (strstr(memory_item->mem_name,"_counters") !=NULL) ||
                   (strstr(memory_item->mem_name,"int_dist_vals") !=NULL)   ) {
          
          hid_t did;
          hsize_t dims[2], start[2], count[2];
	  hsize_t dims_glb[2];
          hid_t sid1;

          dims[0] = npes;
          dims[1] =(hsize_t)memory_item->mem_nelem[0];

	  dims_glb[0] =(hsize_t)memory_item->mem_nelem[0];
	  dims_glb[1] =(hsize_t)memory_item->mem_nelem[1];


          count[0] = 1;
          count[1] = 1;

          start[0] = mype;
          start[1] = 0;
          sid1 = H5Screate_simple (2, dims, NULL);
#ifdef HDF5_FF

           did = H5Dcreate_ff (gid, memory_item->mem_name, filetype, sid1, 
			       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, tid1, e_stack);
	   assert(did > 0);
	   H5Sselect_hyperslab(sid1, H5S_SELECT_SET, start, NULL, count, NULL );
 
	 //   mem_space_id = H5Screate_simple(2, dims_glb, NULL);
// 	   assert(mem_space_id);

	   h5err = H5Dwrite_ff (did, memtype, sid1, sid1, plist_id, mem_ptr, tid1, e_stack);
          
          if(H5Sclose(sid1) < 0)
            printf("HDF5: Could not close dataspace \n");
	  if(H5Dclose_ff(did, H5_EVENT_STACK_NULL) < 0)
             printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);

#else
           did = H5Dcreate2 (gid, memory_item->mem_name, filetype, sid1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
           H5Sselect_hyperslab(sid1, H5S_SELECT_SET, start, NULL, count, NULL ); 
           h5err = H5Dwrite (did, memtype, H5S_ALL, sid1, plist_id, mem_ptr);
          
          if(H5Sclose(sid1) < 0)
            printf("HDF5: Could not close dataspace \n");
           if(H5Dclose(did) < 0)
             printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);
#endif
        } else {

          hid_t did;

#ifdef HDF5_FF
 	  did = H5Dcreate_ff(gid, memory_item->mem_name, filetype, sid,
 				H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, tid1, e_stack);
	  assert(did > 0);
	  h5err = H5Dwrite_ff(did, memtype, H5S_ALL, H5S_ALL, plist_id, mem_ptr, tid1, e_stack);
	  if(H5Dclose_ff(did, H5_EVENT_STACK_NULL) < 0)
	    printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);

// 	  if(0 != mype) {
// 	    H5ESget_count(e_stack, &num_events);
// 	    H5ESwait_all(e_stack, &status);
// 	    H5ESclear(e_stack);
// 	    printf("%d events in event stack. Completion status = %d\n", num_events, status);
// 	    assert(status == H5ES_STATUS_SUCCEED);
//  	  }
	}

	// Barrier to make sure all processes are done writing so Process
	// 0 can finish transaction 1 and acquire a read context on it.
	//MPI_Barrier(MPI_COMM_WORLD);

	// Leader process finished the transaction after all clients
	// finish their updates. Leader also asks the library to acquire
	// the committed transaction, that becomes a readable version
	// after the commit completes.
// 	if(0 == mype) {
// 	  MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);

// 	  // make this synchronous so we know the container version has been acquired
// 	  ret = H5TRfinish(tid1, H5P_DEFAULT, &rid2, H5_EVENT_STACK_NULL);
// 	  assert(0 == ret);
// 	  ret = H5RCrelease(rid2, e_stack);
// 	}
// 	ret = H5TRclose(tid1);
// 	assert(0 == ret);

// 	// release container version 1. This is async.
// 	if(0 == mype) {
// 	  ret = H5RCrelease(rid1, e_stack);
// 	  assert(0 == ret);
// 	}

// 	H5ESget_count(e_stack, &num_events);
// 	H5ESwait_all(e_stack, &status);
// 	printf("%d events in event stack. H5ESwait_all Completion status = %d\n", num_events, status);
// 	H5ESclear(e_stack);
// 	assert(status == H5ES_STATUS_SUCCEED);

#else
	did = H5Dcreate2 (gid, memory_item->mem_name, filetype, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	h5err = H5Dwrite (did, memtype, H5S_ALL, H5S_ALL, plist_id, mem_ptr);
	if(H5Dclose(did) < 0)
	  printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);
        }
#endif
        if(H5Sclose(sid) < 0)
          printf("HDF5: Could not close dataspace \n");

#ifdef HAVE_MPI
        if(H5Pclose(plist_id) < 0)
          printf("HDF5: Could not close property list \n");
#endif
      }
#endif
//       store_field_header(memory_item->mem_name,20);
//       if (memory_item->mem_elsize == 4){
//          store_int_array((int *)mem_ptr, num_elements);
//       } else {
//          store_double_array((double *)mem_ptr, num_elements);
//       }
   }

}

void Crux::store_begin(size_t nsize, int ncycle)
{
#ifdef HDF5_FF
  uint64_t version;
  hid_t    rc_id1, rc_id2, rc_id3, rc_id4, rc_id5, rc_id;
  uint64_t trans_num;
  herr_t ret;
  MPI_Request mpi_req;
  void *token1, *token2, *token3;
  void *gset_token1, *gset_token2, *gset_token3;
  size_t token_size1, token_size2, token_size3;
#endif

#ifdef HAVE_MPI
   int mype;
   MPI_Comm_rank (MPI_COMM_WORLD, &mype );
#endif

   cp_num = checkpoint_counter % num_of_rollback_states;

   cpu_timer_start(&tcheckpoint_time);

   if(crux_type == CRUX_IN_MEMORY){
      if (crux_data[cp_num] != NULL) free(crux_data[cp_num]);
      crux_data[cp_num] = (int *)malloc(nsize);
      crux_data_size[cp_num] = nsize;
      store_fp = fmemopen(crux_data[cp_num], nsize, "w");
   }
   if(crux_type == CRUX_DISK){
      char backup_file[60];

#ifdef HAVE_HDF5
      hid_t plist_id;

      if(USE_HDF5) {

#ifdef HDF5_FF
	sprintf(backup_file,"backup%05d.h5",ncycle);
#else
	sprintf(backup_file,"%s/backup%05d.h5",checkpoint_directory,ncycle);
#endif

        plist_id = H5P_DEFAULT; 
#  ifdef HAVE_MPI
        int mpiInitialized = 0;
        bool phdf5 = false;
        if (MPI_SUCCESS == MPI_Initialized(&mpiInitialized)) {
          phdf5 = true;
        }

        // 
        // Set up file access property list with parallel I/O access
        //
        if( (plist_id = H5Pcreate(H5P_FILE_ACCESS)) < 0)
          printf("HDF5: Could not create property list \n");

#    ifdef HDF5_FF
	H5Pset_fapl_iod(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	// set the metada data integrity checks to happend at transfer through mercury
 	uint32_t cs_scope = 0;
 	cs_scope |= H5_CHECKSUM_TRANSFER;
 	H5Pset_metadata_integrity_scope(plist_id, cs_scope);
#    else
	if( H5Pset_libver_bounds(plist_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST) < 0)
          printf("HDF5: Could set libver bounds \n");
        H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
#    endif
#  endif

#ifdef HDF5_FF

	// create an event Queue for managing asynchronous requests.
	e_stack = H5EScreate();
	assert(e_stack);	

	h5_fid = H5Fcreate_ff(backup_file, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id, H5_EVENT_STACK_NULL );
	assert(h5_fid > 0);
#else
        h5_fid = H5Fcreate(backup_file, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
#endif
        if(!h5_fid){
          printf("HDF5: Could not write HDF5 %s at iteration %d\n",backup_file,ncycle);
        }
#ifdef HAVE_MPI
        if(H5Pclose(plist_id) < 0)
          printf("HDF5: Could not close property list \n");
#endif
#ifdef HDF5_FF

	// Acquire a read handle for container version 1 and create a read context.
	version = 1;
	if ( 0 == mype) {
	  rid1 = H5RCacquire( h5_fid, &version, H5P_DEFAULT, H5_EVENT_STACK_NULL ); 
	} else {
	  rid1 = H5RCcreate( h5_fid, version); 
	}
	assert( rid1 >= 0 ); assert( version == 1 );
	MPI_Barrier( MPI_COMM_WORLD );

	// create transaction object
	tid1 = H5TRcreate(h5_fid, rid1, (uint64_t)2);
	assert(tid1);

	if(0 == mype) {

	  trans_num = 2; 
	  ret = H5TRstart(tid1, H5P_DEFAULT, H5_EVENT_STACK_NULL);
	  assert(0 == ret);

	  // Leader also create some objects in transaction 1

	  // create group hierarchy 

	  if( (h5_gid_c = H5Gcreate_ff(h5_fid, "clamr", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT,tid1, e_stack) ) < 0) 
	    printf("HDF5: Could not create \"clamr\" group \n");
	  if( (h5_gid_m = H5Gcreate_ff(h5_fid, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT,tid1, e_stack) ) < 0)
	    printf("HDF5: Could not create \"mesh\" group \n");
	  if( (h5_gid_s = H5Gcreate_ff(h5_fid, "state", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT,tid1, e_stack) ) < 0)
	    printf("HDF5: Could not create \"state\" group \n");

	  
	  // Get token for group
	  ret = H5Oget_token(h5_gid_c, NULL, &token_size1);
	  assert(0 == ret);
	  ret = H5Oget_token(h5_gid_m, NULL, &token_size2);
	  assert(0 == ret);
	  ret = H5Oget_token(h5_gid_s, NULL, &token_size3);
	  assert(0 == ret);
	  
	  // allocate buffers for each token
	  gset_token1 = malloc(token_size1);
	  gset_token2 = malloc(token_size2);
	  gset_token3 = malloc(token_size3);

	  // get the token buffer
	  ret = H5Oget_token(h5_gid_c, gset_token1, &token_size1);
	  assert(0 == ret);
	  ret = H5Oget_token(h5_gid_m, gset_token2, &token_size2);
	  assert(0 == ret);
	  ret = H5Oget_token(h5_gid_s, gset_token3, &token_size3);
	  assert(0 == ret);

	  // Broadcast token size
	  MPI_Bcast(&token_size1, sizeof(token_size1), MPI_BYTE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(&token_size2, sizeof(token_size2), MPI_BYTE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(&token_size3, sizeof(token_size3), MPI_BYTE, 0, MPI_COMM_WORLD);

	  // Broadcast token
	  MPI_Bcast(gset_token1, token_size1, MPI_BYTE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(gset_token2, token_size2, MPI_BYTE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(gset_token3, token_size3, MPI_BYTE, 0, MPI_COMM_WORLD);

	} 
	else {
	  // Receive token size
	  MPI_Bcast(&token_size1, sizeof(token_size1), MPI_BYTE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(&token_size2, sizeof(token_size2), MPI_BYTE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(&token_size3, sizeof(token_size3), MPI_BYTE, 0, MPI_COMM_WORLD);

	  // Allocate token
	  gset_token1 = malloc(token_size1);
	  gset_token2 = malloc(token_size2);
	  gset_token3 = malloc(token_size3);

	  // Receive token
	  MPI_Bcast(gset_token1, token_size1, MPI_BYTE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(gset_token2, token_size2, MPI_BYTE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(gset_token3, token_size3, MPI_BYTE, 0, MPI_COMM_WORLD);

	  // Open group by token
	  h5_gid_c = H5Oopen_by_token(gset_token1, tid1, e_stack);
	  h5_gid_m = H5Oopen_by_token(gset_token2, tid1, e_stack);
	  h5_gid_s = H5Oopen_by_token(gset_token3, tid1, e_stack);
	}
	free(gset_token1);
	free(gset_token2);
	free(gset_token3);
#else
        if( (h5_gid_c = H5Gcreate(h5_fid, "clamr", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ) < 0) 
          printf("HDF5: Could not create \"clamr\" group \n");
        if( (h5_gid_m = H5Gcreate(h5_fid, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ) < 0)
          printf("HDF5: Could not create \"mesh\" group \n");
        if( (h5_gid_s = H5Gcreate(h5_fid, "state", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ) < 0)
          printf("HDF5: Could not create \"state\" group \n");
#endif
      }

#endif

#ifndef HDF5_FF
//       sprintf(backup_file,"backup%05d.crx",ncycle);
//       store_fp = fopen(backup_file,"w");
//       if(!store_fp){
//          printf("Could not write %s at iteration %d\n",backup_file,ncycle);
//       }

//       char symlink_file[60];
//       sprintf(symlink_file,"%s/backup%1d.crx",checkpoint_directory,cp_num);
//       int ireturn = symlink(backup_file, symlink_file);
//    if (ireturn == -1) {
//       printf("Warning: error returned with symlink call for file %s and symlink %s\n",
//              backup_file,symlink_file);
//    }
#endif
   }

   if (do_crux_timing){
      checkpoint_timing_size += nsize;
   }
}

void Crux::store_field_header(const char *name, int name_size){
   assert(name != NULL && store_fp != NULL);
   fwrite(name,sizeof(char),name_size,store_fp);
}

void Crux::store_bools(bool *bool_vals, size_t nelem)
{
   assert(bool_vals != NULL && store_fp != NULL);
   fwrite(bool_vals,sizeof(bool),nelem,store_fp);
}

void Crux::store_ints(int *int_vals, size_t nelem)
{
   assert(int_vals != NULL && store_fp != NULL);
   fwrite(int_vals,sizeof(int),nelem,store_fp);
}

void Crux::store_longs(long long *long_vals, size_t nelem)
{
   assert(long_vals != NULL && store_fp != NULL);
   fwrite(long_vals,sizeof(long long),nelem,store_fp);
}

void Crux::store_sizets(size_t *size_t_vals, size_t nelem)
{
  //   assert(size_t_vals != NULL && store_fp != NULL);
  // fwrite(size_t_vals,sizeof(size_t),nelem,store_fp);
}

void Crux::store_doubles(double *double_vals, size_t nelem)
{
   assert(double_vals != NULL && store_fp != NULL);
   fwrite(double_vals,sizeof(double),nelem,store_fp);
}

void Crux::store_int_array(int *int_array, size_t nelem)
{
   assert(int_array != NULL && store_fp != NULL);
   fwrite(int_array,sizeof(int),nelem,store_fp);
}

void Crux::store_long_array(long long *long_array, size_t nelem)
{
   assert(long_array != NULL && store_fp != NULL);
   fwrite(long_array,sizeof(long long),nelem,store_fp);
}

void Crux::store_float_array(float *float_array, size_t nelem)
{
   assert(float_array != NULL && store_fp != NULL);
   fwrite(float_array,sizeof(float),nelem,store_fp);
}

void Crux::store_double_array(double *double_array, size_t nelem)
{
   assert(double_array != NULL && store_fp != NULL);
   fwrite(double_array,sizeof(double),nelem,store_fp);
}

void Crux::store_end(void)
{
//    assert(store_fp != NULL);
//    fclose(store_fp);

#ifdef HAVE_MPI
   int mype;
   MPI_Comm_rank (MPI_COMM_WORLD, &mype );
#endif

#ifdef HAVE_HDF5
   
   if(USE_HDF5) {

#    ifdef HDF5_FF

     H5ES_status_t status;
     size_t num_events = 0;
     herr_t ret;
     uint64_t version;

     /* none leader procs have to complete operations before notifying the leader */
    if(0 != mype) {
        H5ESget_count(e_stack, &num_events);
        H5ESwait_all(e_stack, &status);
        H5ESclear(e_stack);
        printf("%d events in event stack. Completion status = %d\n", num_events, status);
        assert(status == H5ES_STATUS_SUCCEED);
    }

    /* Barrier to make sure all processes are done writing so Process
       0 can finish transaction 1 and acquire a read context on it. */
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(0 == mype) {
      //  MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);

      // make this synchronous so we know the container version has been acquired
      ret = H5TRfinish(tid1, H5P_DEFAULT, &rid2, H5_EVENT_STACK_NULL);
      assert(0 == ret);
    }

    ret = H5TRclose(tid1);
    assert(0 == ret);
    /* release container version 1. This is async. */
    if(0 == mype) {
        ret = H5RCrelease(rid1, e_stack);
        assert(0 == ret);
    }
    H5ESget_count(e_stack, &num_events);
    H5ESwait_all(e_stack, &status);
    printf("%d events in event stack. H5ESwait_all Completion status = %d\n", num_events, status);
    H5ESclear(e_stack);
    assert(status == H5ES_STATUS_SUCCEED);

    if(H5Gclose_ff(h5_gid_c, e_stack) < 0)
      printf("HDF5: Could not close clamr group \n");
    if(H5Gclose_ff(h5_gid_m, e_stack) < 0)
      printf("HDF5: Could not close mesh group \n");
    if(H5Gclose_ff(h5_gid_s, e_stack) < 0)
      printf("HDF5: Could not close state group \n");

    /* Tell other procs that container version 2 is acquired */
    version = 2;
    MPI_Bcast(&version, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    /* other processes just create a read context object; no need to
       acquire it */
    if(0 != mype) {
      rid2 = H5RCcreate(h5_fid, version);
      assert(rid2 > 0);
    }
    /* close some objects */

    H5ESwait(e_stack, 0, &status);

    H5ESget_count(e_stack, &num_events);
    H5ESwait_all(e_stack, &status);
    printf("%d events in event stack. H5ESwait_all Completion status = %d\n", 
           num_events, status);
    H5ESclear(e_stack);

    MPI_Barrier(MPI_COMM_WORLD);    
    if(mype == 0) {
        ret = H5RCrelease(rid2, e_stack);
        assert(0 == ret);
    }


    if(H5Fclose_ff(h5_fid, 1, H5_EVENT_STACK_NULL) != 0)
      printf("HDF5: Could not close HDF5 file \n");

      H5ESget_count(e_stack, &num_events);

      H5EStest_all(e_stack, &status);
      printf("%d events in event stack. H5EStest_all Completion status = %d\n", num_events, status);
      
      H5ESwait_all(e_stack, &status);
      printf("%d events in event stack. H5ESwait_all Completion status = %d\n", num_events, status);
      
      ret = H5RCclose(rid1);
      assert(0 == ret);
      ret = H5RCclose(rid2);
      assert(0 == ret);
      
      H5ESclear(e_stack);
      
      ret = H5ESclose(e_stack);
      assert(ret == 0);  
      

#    else
     if(H5Gclose(h5_gid_c) < 0)
       printf("HDF5: Could not close clamr group \n");
     if(H5Gclose(h5_gid_m) < 0)
       printf("HDF5: Could not close mesh group \n");
     if(H5Gclose(h5_gid_s) < 0)
       printf("HDF5: Could not close state group \n");
     if(H5Fclose(h5_fid) != 0) {
       printf("HDF5: Could not close HDF5 file \n");
     }
#    endif
   }
#endif

   double checkpoint_total_time = cpu_timer_stop(tcheckpoint_time);

   if (do_crux_timing){
      fprintf(crux_time_fp, "Total time for checkpointing was %g seconds\n", checkpoint_total_time);
      checkpoint_timing_count++;
      checkpoint_timing_sum += checkpoint_total_time;
   }

   checkpoint_counter++;
}

int restore_type = RESTORE_NONE;

void Crux::restore_MallocPlus(MallocPlus memory){
   char test_name[24];
   malloc_plus_memory_entry *memory_item;
   for (memory_item = memory.memory_entry_by_name_begin(); 
      memory_item != memory.memory_entry_by_name_end();
      memory_item = memory.memory_entry_by_name_next() ){

      void *mem_ptr = memory_item->mem_ptr;
      if ((memory_item->mem_flags & RESTART_DATA) == 0) continue;

      int num_elements = 1;
      for (uint i = 1; i < memory_item->mem_ndims; i++){
	 num_elements *= memory_item->mem_nelem[i];
      }

      if (DEBUG) {
        printf("MallocPlus ptr  %p: name %10s ptr %p dims %lu nelem (",
           mem_ptr,memory_item->mem_name,memory_item->mem_ptr,memory_item->mem_ndims);

        char nelemstring[80];
        char *str_ptr = nelemstring;
        str_ptr += sprintf(str_ptr,"%lu", memory_item->mem_nelem[0]);
        for (uint i = 1; i < memory_item->mem_ndims; i++){
           str_ptr += sprintf(str_ptr,", %lu", memory_item->mem_nelem[i]);
        }
        printf("%12s",nelemstring);

        printf(") elsize %lu flags %d capacity %lu\n",
           memory_item->mem_elsize,memory_item->mem_flags,memory_item->mem_capacity);
      }

      restore_field_header(test_name,20);
      if (strcmp(test_name,memory_item->mem_name) != 0) {
         printf("ERROR in restore checkpoint for %s %s\n",test_name,memory_item->mem_name);
      }

      if (memory_item->mem_elsize == 4){
         restore_int_array((int *)mem_ptr, num_elements);
      } else {
         restore_double_array((double *)mem_ptr, num_elements);
      }
   }
}

void Crux::restore_begin(char *restart_file, int rollback_counter)
{
   rs_num = rollback_counter % num_of_rollback_states;

   cpu_timer_start(&trestore_time);

   if (restart_file != NULL){
      printf("\n  ================================================================\n");
      printf(  "  Restoring state from disk file %s\n",restart_file);
      printf(  "  ================================================================\n\n");
      restore_fp = fopen(restart_file,"r");
      if(!restore_fp){
         //printf("Could not write %s at iteration %d\n",restart_file,crux_int_vals[8]);
         printf("Could not open restart file %s\n",restart_file);
      }
      restore_type = RESTORE_RESTART;
   } else if(crux_type == CRUX_IN_MEMORY){
      printf("Restoring state from memory rollback number %d rollback_counter %d\n",rs_num,rollback_counter);
      restore_fp = fmemopen(crux_data[rs_num], crux_data_size[rs_num], "r");
      restore_type = RESTORE_ROLLBACK;
   } else if(crux_type == CRUX_DISK){
      char backup_file[60];

      sprintf(backup_file,"%s/backup%d.crx",checkpoint_directory,rs_num);
      printf("Restoring state from disk file %s rollback_counter %d\n",backup_file,rollback_counter);
      restore_fp = fopen(backup_file,"r");
      if(!restore_fp){
         //printf("Could not write %s at iteration %d\n",backup_file,crux_int_vals[8]);
         printf("Could not open restore file %s\n",backup_file);
      }
      restore_type = RESTORE_ROLLBACK;
   }
}

void Crux::restore_field_header(char *name, int name_size)
{
   int name_read = fread(name,sizeof(char),name_size,restore_fp);
   if (name_read != name_size){
      printf("Warning: number of elements read %d is not equal to request %d\n",name_read,name_size);
   }
}

void Crux::restore_bools(bool *bool_vals, size_t nelem)
{
   size_t nelem_read = fread(bool_vals,sizeof(bool),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

void Crux::restore_ints(int *int_vals, size_t nelem)
{
   size_t nelem_read = fread(int_vals,sizeof(int),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

void Crux::restore_longs(long long *long_vals, size_t nelem)
{
   size_t nelem_read = fread(long_vals,sizeof(long),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

void Crux::restore_sizets(size_t *size_t_vals, size_t nelem)
{
   size_t nelem_read = fread(size_t_vals,sizeof(size_t),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

void Crux::restore_doubles(double *double_vals, size_t nelem)
{
   size_t nelem_read = fread(double_vals,sizeof(double),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

int *Crux::restore_int_array(int *int_array, size_t nelem)
{
   size_t nelem_read = fread(int_array,sizeof(int),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
   return(int_array);
}

long long *Crux::restore_long_array(long long *long_array, size_t nelem)
{
   size_t nelem_read = fread(long_array,sizeof(long long),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
   return(long_array);
}

float *Crux::restore_float_array(float *float_array, size_t nelem)
{
   size_t nelem_read = fread(float_array,sizeof(float),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
   return(float_array);
}

double *Crux::restore_double_array(double *double_array, size_t nelem)
{
   size_t nelem_read = fread(double_array,sizeof(double),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
   return(double_array);
}

void Crux::restore_end(void)
{
   double restore_total_time = cpu_timer_stop(trestore_time);

   if (do_crux_timing){
      if (restore_type == RESTORE_RESTART) {
         fprintf(crux_time_fp, "Total time for restore was %g seconds\n", restore_total_time);
      } else if (restore_type == RESTORE_ROLLBACK){
         fprintf(crux_time_fp, "Total time for rollback %d was %g seconds\n", rollback_attempt, restore_total_time);
      }
   }

   fclose(restore_fp);

}

int Crux::get_rollback_number()
{
  rollback_attempt++;
  return(checkpoint_counter % num_of_rollback_states);
}
