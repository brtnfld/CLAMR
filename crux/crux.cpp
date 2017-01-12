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
int verbose_io = 0;        // Verbose (print CV contents between every transaction) defaults to no

#define RESTORE_NONE     0
#define RESTORE_RESTART  1
#define RESTORE_ROLLBACK 2

#ifndef DEBUG
#define DEBUG 0
#endif
#undef DEBUG_RESTORE_VALS

using namespace std;
using PP::PowerParser;
// Pointers to the various objects.
PowerParser *parse;

char checkpoint_directory[] = "checkpoint_output";
int cp_num, rs_num;
int *backup;
void **crux_data;
size_t *crux_data_size;
bool USE_HDF5 = true; //MSB fix thimake global

#ifdef HAVE_HDF5
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
FILE *store_fp, *restore_fp;
#ifdef HAVE_MPI
static MPI_File mpi_store_fp, mpi_restore_fp;
#endif
static int npes = 1;

char backup_file[60];

#ifdef HDF5_FF
void
print_container_contents_ff( hid_t file_id, hid_t rc_id, int my_rank );
#endif
#ifdef HAVE_HDF5
void
print_container_contents( hid_t file_id, int my_rank );
#endif
static int mype = 0;


#ifdef HDF5_FF
void
print_container_contents_ff( hid_t file_id, hid_t rc_id, int my_rank );
#else
void
print_container_contents( hid_t file_id, int my_rank );
#endif



Crux::Crux(int crux_type_in, int num_of_rollback_states_in, bool restart)
{
#ifdef HAVE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD,&mype);
#endif

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

    int num_elements = 1;
    for (uint i = 0; i < memory_item->mem_ndims; i++){
      num_elements *= memory_item->mem_nelem[i];
    }

    size_t numcells_glb;
    MPI_Allreduce(&memory_item->mem_nelem[0], &numcells_glb, 1, MPI_UINT64_T, MPI_SUM,
		  MPI_COMM_WORLD);
    
    uint64_t *rncells; // All cell counts from each task
    
    // Gather ncells on each processor
    rncells = (uint64_t *) malloc(npes*sizeof(uint64_t));

    MPI_Allgather(&memory_item->mem_nelem[0], 1, MPI_UNSIGNED_LONG_LONG, rncells, 1, MPI_LONG_LONG, MPI_COMM_WORLD);

    printf("CRUX: %s %ld %ld \n",memory_item->mem_name, memory_item->mem_ncells_global,memory_item->mem_noffset);

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
    hid_t memspace, filespace;
    hsize_t dims[2], start[2], count[2];

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
	if( (strncmp(memory_item->mem_name,"state_long_vals",15) == 0) ) {
	  filetype = H5T_NATIVE_LONG;
	  memtype = H5T_NATIVE_LONG;
	} else if(h5_spoutput) {
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
#  ifdef HAVE_MPI
        //
        // Create property list for collective dataset write.
        //
      plist_id = H5Pcreate(H5P_DATASET_XFER);
      
#    ifdef HDF5_FF

	tid1 = H5TRcreate(h5_fid, rid2, (uint64_t)2);

	trans_num = 3;
	printf("H5TRstart 3\n");
	ret = H5TRstart(tid1, H5P_DEFAULT, H5_EVENT_STACK_NULL);
#    else
      H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#    endif

#  endif
      if( (strncmp(memory_item->mem_name,"state_long_vals",15) == 0) ||
	  (strncmp(memory_item->mem_name,"mesh_double_vals",16) == 0) ) {
	hid_t aid;
#  ifdef HDF5_FF
	if(mype == 0) {
	  aid = H5Acreate_ff(gid, memory_item->mem_name, filetype, sid, 
			     H5P_DEFAULT, H5P_DEFAULT, tid1, e_stack);
	  assert(aid > 0);
          // Write the attribute data.
	  h5err = H5Awrite_ff(aid, filetype, mem_ptr, tid1, e_stack);
	  h5err = H5Aclose_ff(aid, e_stack);
	}
#  else
	aid = H5Acreate2(gid, memory_item->mem_name, filetype, sid, H5P_DEFAULT, H5P_DEFAULT);
	// Write the attribute data.
	h5err = H5Awrite(aid, filetype, mem_ptr);
	h5err = H5Aclose(aid);
#  endif
      } else if( (strstr(memory_item->mem_name,"_timer") !=NULL) ||
		 (strstr(memory_item->mem_name,"_counters") !=NULL) ||
		 (strstr(memory_item->mem_name,"int_dist_vals") !=NULL)   ) {
          
	hid_t did;
	hsize_t dims[2], start[2], count[2]; // MSB
	hsize_t dims_glb[2];
	hid_t sid1;
	hid_t memspace, filespace;
	
	dims[0] = npes;
	dims[1] = (hsize_t)memory_item->mem_nelem[0];
	
	dims_glb[0] =(hsize_t)memory_item->mem_nelem[0];
	dims_glb[1] =(hsize_t)memory_item->mem_nelem[1];
	
	
	count[0] = 1;
	count[1] = (hsize_t)memory_item->mem_nelem[0];
	
	start[0] = mype;
	start[1] = 0;

	filespace = H5Screate_simple (2, dims, NULL);
	memspace = H5Screate_simple(2, count, NULL); 
#  ifdef HDF5_FF

	did = H5Dcreate_ff (gid, memory_item->mem_name, filetype, filespace, 
			      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, tid1, e_stack);
	assert(did > 0);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL ); 
	h5err = H5Dwrite_ff (did, memtype, memspace, filespace, plist_id, mem_ptr, tid1, e_stack);
        
	if(H5Sclose(filespace) < 0)
	  printf("HDF5: Could not close dataspace \n");
	if(H5Sclose(memspace) < 0)
	  printf("HDF5: Could not close dataspace \n");
	if(H5Dclose_ff(did, H5_EVENT_STACK_NULL) < 0)
	  printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);

#  else
	int *val = (int*)mem_ptr;
	printf("%s %d %d \n",memory_item->mem_name,val[0], val[1]);
	did = H5Dcreate2 (gid, memory_item->mem_name, filetype, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL ); 
	h5err = H5Dwrite (did, memtype, memspace, filespace, plist_id, mem_ptr);
        
	if(H5Sclose(filespace) < 0)
	  printf("HDF5: Could not close dataspace \n");
	if(H5Sclose(memspace) < 0)
	  printf("HDF5: Could not close dataspace \n");
	if(H5Dclose(did) < 0)
	  printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);

#  endif
      } else if( (strstr(memory_item->mem_name,"bootstrap_") !=NULL) ) {

	hid_t did;

#  ifdef HDF5_FF

	did = H5Dcreate_ff (gid, memory_item->mem_name, filetype, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, tid1, e_stack);
	if(mype == 0) {
	  h5err = H5Dwrite_ff (did, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, mem_ptr, tid1, e_stack);
	}
	  
	if(H5Dclose_ff(did, H5_EVENT_STACK_NULL) < 0)
	  printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);
#  else

	did = H5Dcreate2 (gid, memory_item->mem_name, filetype, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(mype == 0) {
	  h5err = H5Dwrite (did, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, mem_ptr);
	}
	  
	if(H5Dclose(did) < 0)
	  printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);
#  endif

      } else {
	
	hid_t filespace;
	hid_t memspace;
	hsize_t dims[1], start[1], count[1];
#  ifdef HAVE_MPI
	  
	dims[0] = (hsize_t)numcells_glb;
	count[0] = (hsize_t)memory_item->mem_nelem[0];
	start[0] = 0;
	for (int j=0; j<mype; j++)
	  start[0] = start[0] + rncells[j];
	  
	  // printf("start %s %ld \n", memory_item->mem_name, start[0]);
          // MSB check this is not needed: start[0] = (hsize_t)(count[0]*mype);
	  //  start[0] = (hsize_t)(count[0]*mype);
	  
	filespace = H5Screate_simple (1, dims, NULL);
	
	memspace = H5Screate_simple(1, count, NULL);
#  else  
	memspace = H5S_ALL;
	filespace = H5S_ALL;
#  endif
	  
	hid_t did;

#  ifdef HDF5_FF
	did = H5Dcreate_ff(gid, memory_item->mem_name, filetype, filespace,
			   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, tid1, e_stack);
	assert(did > 0);
	
	// Select hyperslab in the file.
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL );
	h5err = H5Dwrite_ff (did, memtype, memspace, filespace, plist_id, mem_ptr, tid1, e_stack);
	
	if(H5Dclose_ff(did, H5_EVENT_STACK_NULL) < 0)
	  printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);

	h5err = H5Sclose(filespace);
	h5err = H5Sclose(memspace);

	  if(0 != mype) {
	    H5ESget_count(e_stack, &num_events);
	    H5ESwait_all(e_stack, &status);
	    H5ESclear(e_stack);
	    printf("%d events in event stack. Completion status = %d\n", num_events, status);
	    assert(status == H5ES_STATUS_SUCCEED);
 	  }

	// Barrier to make sure all processes are done writing so Process
	// 0 can finish transaction 1 and acquire a read context on it.
	//MPI_Barrier(MPI_COMM_WORLD);

	// Leader process finished the transaction after all clients
	// finish their updates. Leader also asks the library to acquire
	// the committed transaction, that becomes a readable version
	// after the commit completes.
	if(0 == mype) {
	  MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);

	  // make this synchronous so we know the container version has been acquired
	  ret = H5TRfinish(tid1, H5P_DEFAULT, &rid2, H5_EVENT_STACK_NULL);
	  assert(0 == ret);
	  ret = H5RCrelease(rid2, e_stack);
	}
	ret = H5TRclose(tid1);
	assert(0 == ret);

	// release container version 1. This is async.
	if(0 == mype) {
	  ret = H5RCrelease(rid1, e_stack);
	  assert(0 == ret);
	}

	H5ESget_count(e_stack, &num_events);
	H5ESwait_all(e_stack, &status);
	printf("%d events in event stack. H5ESwait_all Completion status = %d\n", num_events, status);
	H5ESclear(e_stack);
	assert(status == H5ES_STATUS_SUCCEED);

#  else

      int *val = (int*)mem_ptr;
      printf("THIS %s %d %d \n",memory_item->mem_name,val[0], val[1]);
      did = H5Dcreate2(gid, memory_item->mem_name, filetype, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      // Select hyperslab in the file.
      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL );
      
      h5err = H5Dwrite (did, memtype, memspace, filespace, plist_id, mem_ptr);
      if(H5Dclose(did) < 0)
	printf("HDF5: Could not close dataset %s \n",memory_item->mem_name);
      
#    ifdef HAVE_MPI
      h5err = H5Sclose(filespace);
      h5err = H5Sclose(memspace);
#    endif
     
#  endif 
      }
      if(H5Sclose(sid) < 0)
	printf("HDF5: Could not close dataspace \n");
#  ifdef HAVE_MPI
    if(H5Pclose(plist_id) < 0)
      printf("HDF5: Could not close property list \n");
#  endif
     }
#endif
    if (USE_HDF5 != true) {
      store_field_header(memory_item->mem_name,30);
      if (memory_item->mem_flags & REPLICATED_DATA) { 
	if (memory_item->mem_elsize == 4){
	  store_replicated_int_array((int *)mem_ptr, num_elements);
	} else {
	  store_replicated_double_array((double *)mem_ptr, num_elements);
	}
      } else {
	if (memory_item->mem_elsize == 4){
	  store_int_array((int *)mem_ptr, num_elements);
	} else {
	  store_double_array((double *)mem_ptr, num_elements);
	}
      }
    }
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
  size_t token_size[3];
  void *gset_token[3];
  int dims_token[3];
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
     // MSB  char backup_file[60];

#ifdef HAVE_HDF5
      hid_t plist_id;

      if(USE_HDF5) {

#  ifdef HDF5_FF
	sprintf(backup_file,"backup%05d.h5",ncycle);
#  else
	sprintf(backup_file,"%s/backup%05d.h5",checkpoint_directory,ncycle);
#  endif

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

#      ifdef HDF5_FF
	H5Pset_fapl_iod(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	// set the metada data integrity checks to happend at transfer through mercury
//  	uint32_t cs_scope = 0;
//  	cs_scope |= H5_CHECKSUM_TRANSFER;
//  	H5Pset_metadata_integrity_scope(plist_id, cs_scope);
#      else
	if( H5Pset_libver_bounds(plist_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST) < 0)
          printf("HDF5: Could set libver bounds \n");
        H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
#      endif
#  endif

#  ifdef HDF5_FF

	// create an event Queue for managing asynchronous requests.
	e_stack = H5EScreate();
	assert(e_stack);	

	h5_fid = H5Fcreate_ff(backup_file, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id, H5_EVENT_STACK_NULL );
	assert(h5_fid > 0);
#  else
        h5_fid = H5Fcreate(backup_file, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
#  endif
        if(!h5_fid){
          printf("HDF5: Could not write HDF5 %s at iteration %d\n",backup_file,ncycle);
        }
#  ifdef HAVE_MPI
        if(H5Pclose(plist_id) < 0)
          printf("HDF5: Could not close property list \n");
#  endif
#  ifdef HDF5_FF

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
	  ret = H5Oget_token(h5_gid_c, NULL, &token_size[0]);
	  assert(0 == ret);
	  ret = H5Oget_token(h5_gid_m, NULL, &token_size[1]);
	  assert(0 == ret);
	  ret = H5Oget_token(h5_gid_s, NULL, &token_size[2]);
	  assert(0 == ret);
	  
	  // allocate buffers for each token

	  gset_token[0] = malloc(token_size[0]);
	  gset_token[1] = malloc(token_size[1]);
	  gset_token[2] = malloc(token_size[2]);

	  // get the token buffer

	  ret = H5Oget_token(h5_gid_c, gset_token[0], &token_size[0]);
	  assert(0 == ret);
	  ret = H5Oget_token(h5_gid_m, gset_token[1], &token_size[1]);
	  assert(0 == ret);
	  ret = H5Oget_token(h5_gid_s, gset_token[2], &token_size[2]);
	  assert(0 == ret);

	  // Broadcast token size

	  MPI_Bcast(&token_size, sizeof(size_t)*3, MPI_BYTE, 0, MPI_COMM_WORLD);

	  // Broadcast token
	  //MPI_Bcast(gset_token, token_size[0]+token_size[1]+token_size[2], MPI_BYTE, 0, MPI_COMM_WORLD);

 	  MPI_Bcast(gset_token[0], token_size[0], MPI_BYTE, 0, MPI_COMM_WORLD);
  	  MPI_Bcast(gset_token[1], token_size[1], MPI_BYTE, 0, MPI_COMM_WORLD);
  	  MPI_Bcast(gset_token[2], token_size[2], MPI_BYTE, 0, MPI_COMM_WORLD);

	} 
	else {
	  // Receive token size

	  MPI_Bcast(&token_size, sizeof(size_t)*3, MPI_BYTE, 0, MPI_COMM_WORLD);

	  // Allocate token

	  gset_token[0] = malloc(token_size[0]);
	  gset_token[1] = malloc(token_size[1]);
	  gset_token[2] = malloc(token_size[2]);

	  // Receive token

	  //MPI_Bcast(gset_token, token_size[0]+token_size[1]+token_size[2], MPI_BYTE, 0, MPI_COMM_WORLD);


 	  MPI_Bcast(gset_token[0], token_size[0], MPI_BYTE, 0, MPI_COMM_WORLD);
 	  MPI_Bcast(gset_token[1], token_size[1], MPI_BYTE, 0, MPI_COMM_WORLD);
 	  MPI_Bcast(gset_token[2], token_size[2], MPI_BYTE, 0, MPI_COMM_WORLD);


	  // Open group by token
 	  h5_gid_c = H5Oopen_by_token(gset_token[0], tid1, e_stack);
 	  h5_gid_m = H5Oopen_by_token(gset_token[1], tid1, e_stack);
 	  h5_gid_s = H5Oopen_by_token(gset_token[2], tid1, e_stack);
	}
 	free(gset_token[0]);
 	free(gset_token[1]);
 	free(gset_token[2]);
#  else
        if( (h5_gid_c = H5Gcreate(h5_fid, "clamr", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ) < 0) 
          printf("HDF5: Could not create \"clamr\" group \n");
        if( (h5_gid_m = H5Gcreate(h5_fid, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ) < 0)
          printf("HDF5: Could not create \"mesh\" group \n");
        if( (h5_gid_s = H5Gcreate(h5_fid, "state", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ) < 0)
          printf("HDF5: Could not create \"state\" group \n");
#  endif
#endif
      if (USE_HDF5 != true) {
	sprintf(backup_file,"%s/backup%05d.crx",checkpoint_directory,ncycle);
#ifdef HAVE_MPI
	int iret = MPI_File_open(MPI_COMM_WORLD, backup_file, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_store_fp);
	if(iret != MPI_SUCCESS) {
	  printf("Could not write %s at iteration %d\n",backup_file,ncycle);
	}
#else
	store_fp = fopen(backup_file,"w");
	if(!store_fp){
	  printf("Could not write %s at iteration %d\n",backup_file,ncycle);
	}
#endif

	if (mype == 0) {
	  char symlink_file[60];
	  sprintf(symlink_file,"%s/backup%1d.crx",checkpoint_directory,cp_num);
	  symlink(backup_file, symlink_file);
//      int ireturn = symlink(backup_file, symlink_file);
//      if (ireturn == -1) {
//         printf("Warning: error returned with symlink call for file %s and symlink %s\n",
//                backup_file,symlink_file);
//      }
	}
      }
   }

   if (do_crux_timing){
      checkpoint_timing_size += nsize;
   }
   }
}

void Crux::store_field_header(const char *name, int name_size){
#ifdef HAVE_MPI
   assert(name != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, (void *)name, name_size, MPI_CHAR, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_CHAR, &count);
   printf("Wrote %d characters at line %d in file %s\n",count,__LINE__,__FILE__);
#endif

#else
   assert(name != NULL && store_fp != NULL);
   fwrite(name,sizeof(char),name_size,store_fp);
#endif
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
#ifdef HAVE_MPI
   assert(int_array != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, int_array, (int)nelem, MPI_INT, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("Wrote %d integers at line %d in file %s\n",count,__LINE__,__FILE__);
#endif

#else
   assert(int_array != NULL && store_fp != NULL);
   fwrite(int_array,sizeof(int),nelem,store_fp);
#endif
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
#ifdef HAVE_MPI
   assert(double_array != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, double_array, (int)nelem, MPI_DOUBLE, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE, &count);
   printf("Wrote %d doubles at line %d in file %s\n",count,__LINE__,__FILE__);
#endif

#else
   assert(double_array != NULL && store_fp != NULL);
   fwrite(double_array,sizeof(double),nelem,store_fp);
#endif
}

void Crux::store_replicated_int_array(int *int_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(int_array != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, int_array, (int)nelem, MPI_INT, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("Wrote %d integers at line %d in file %s\n",count,__LINE__,__FILE__);
#endif

#else
   assert(int_array != NULL && store_fp != NULL);
   fwrite(int_array,sizeof(int),nelem,store_fp);
#endif
}

void Crux::store_replicated_double_array(double *double_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(double_array != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, double_array, (int)nelem, MPI_DOUBLE, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE, &count);
   printf("Wrote %d doubles at line %d in file %s\n",count,__LINE__,__FILE__);
#endif

#else
   assert(double_array != NULL && store_fp != NULL);
   fwrite(double_array,sizeof(double),nelem,store_fp);
#endif
}

void Crux::store_end(void)
{
//    assert(store_fp != NULL);
//    fclose(store_fp);

#ifdef HAVE_MPI
   int mype;
   MPI_Comm_rank (MPI_COMM_WORLD, &mype );
   if(!USE_HDF5)
     MPI_File_close(&mpi_store_fp);
#else
   assert(store_fp != NULL);
   fclose(store_fp);
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

#ifdef HDF5_FF

   // Now, reopen file and show contents of all available CVs

   hid_t last_rc_id;
   uint64_t last_version, v, version;
   hid_t file_id, fapl_id, rc_id;

   /* Reopen the file Read/Write */
   fprintf( stderr, "%d: open the container\n", mype );

   fapl_id = H5Pcreate( H5P_FILE_ACCESS );
   H5Pset_fapl_iod( fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL );

   /* Get latest CV on open */
   file_id = H5Fopen_ff(backup_file, H5F_ACC_RDONLY, fapl_id, &last_rc_id, H5_EVENT_STACK_NULL ); 

   H5RCget_version( last_rc_id, &last_version );

   v = last_version;
   for ( v; v <= last_version; v++ ) {
     version = v;
     if ( 0 == mype ) {
       if(v < last_version) {
	 fprintf( stderr, "r%d: Try to acquire read context for cv %d\n", mype, (int)version );
	 rc_id = H5RCacquire( file_id, &version, H5P_DEFAULT, H5_EVENT_STACK_NULL );
       }
       else
	 rc_id = last_rc_id;
     }

     MPI_Bcast( &rc_id, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );

     if ( rc_id < 0 ) {
       if ( 0 == mype ) {
	 fprintf( stderr, "r%d: Failed to acquire read context for cv %d\n", mype, (int)v );
       }
     } else {
       if ( 0 != mype ) {
	 rc_id = H5RCcreate( file_id, version ); 
       }
       assert ( version == v );
       print_container_contents_ff(file_id, rc_id, mype );

       MPI_Barrier( MPI_COMM_WORLD );
       if(v < last_version) {
	 if ( 0 == mype ) {
	   H5RCrelease( rc_id, H5_EVENT_STACK_NULL ); 
	 }
	 H5RCclose( rc_id ); 
       }
     }
   }

   /* Release the read handle and close read context on cv obtained from H5Fopen_ff (by all ranks) */
   H5RCrelease( last_rc_id, H5_EVENT_STACK_NULL );
   H5RCclose( last_rc_id );

   /* Close 2 H5 Objects that are still open */
   fprintf( stderr, "r%d: close all h5 objects that are still open\n", mype );
   H5Fclose_ff( file_id, 1, H5_EVENT_STACK_NULL );
   H5Pclose( fapl_id );

#endif

#ifdef HAVE_HDF5
   if(USE_HDF5) {
   // Now, reopen file and show contents of all available CVs

     hid_t file_id, fapl_id;

     /* Reopen the file Read/Write */
     fprintf( stderr, "%d: open the file\n", mype );

     fapl_id = H5Pcreate( H5P_FILE_ACCESS );
     H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);

   /* Get latest CV on open */
     file_id = H5Fopen(backup_file, H5F_ACC_RDONLY, fapl_id); 

     print_container_contents(file_id, mype );
     
     MPI_Barrier( MPI_COMM_WORLD );
     
   /* Close 2 H5 Objects that are still open */
     H5Fclose( file_id );
     H5Pclose( fapl_id );
   }

#endif

}

int restore_type = RESTORE_NONE;

void Crux::restore_MallocPlus(MallocPlus memory){
      printf("Crux::restore_MallocPlus \n");
   char test_name[34];
#ifdef HAVE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD,&mype);
   MPI_Comm_size(MPI_COMM_WORLD,&npes);
#endif

   malloc_plus_memory_entry *memory_item;
   for (memory_item = memory.memory_entry_by_name_begin(); 
      memory_item != memory.memory_entry_by_name_end();
      memory_item = memory.memory_entry_by_name_next() ){

      void *mem_ptr = memory_item->mem_ptr;
      if ((memory_item->mem_flags & RESTART_DATA) == 0) continue;

      int num_elements = 1;
      for (uint i = 0; i < memory_item->mem_ndims; i++){
	 num_elements *= memory_item->mem_nelem[i];
      }
      //if (DEBUG) {
      if (1) {
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

      if(!USE_HDF5) {
	restore_field_header(test_name,30);
	if (strcmp(test_name,memory_item->mem_name) != 0) {
	  printf("ERROR in restore checkpoint for %s %s\n",test_name,memory_item->mem_name);
	}

	if (memory_item->mem_flags & REPLICATED_DATA) { 
	  if (memory_item->mem_elsize == 4){
            restore_replicated_int_array((int *)mem_ptr, num_elements);
	  } else {
            restore_replicated_double_array((double *)mem_ptr, num_elements);
	  }
	} else {
	  if (memory_item->mem_elsize == 4){
            restore_int_array((int *)mem_ptr, num_elements);
	  } else {
            restore_double_array((double *)mem_ptr, num_elements);
	  }
	}
      }
#ifdef HAVE_HDF5
      else {
	if(strstr(memory_item->mem_name,"boot") != NULL) {
	  hid_t gid, dset_id;
	  gid = H5Gopen(h5_fid,"clamr", H5P_DEFAULT);
	  if( (dset_id = H5Dopen(gid, memory_item->mem_name, H5P_DEFAULT)) < 0) {
	    printf("ERROR in restore checkpoint for clamr/%s\n",memory_item->mem_name);
	  }
	  hid_t dtype = H5Dget_type(dset_id);
	  H5Dread( dset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, mem_ptr);
	  H5Tclose(dtype);
	  H5Gclose(gid);
	  H5Dclose(dset_id);
	} else if( strcmp(memory_item->mem_name,"i")  || 
		   strcmp(memory_item->mem_name,"j")  || 
		   strcmp(memory_item->mem_name,"level")  ||
		   strcmp(memory_item->mem_name,"amesh_int_dist_vals") ) {
	  hid_t gid, dset_id;
	  gid = H5Gopen(h5_fid,"mesh", H5P_DEFAULT);
	  if( (dset_id = H5Dopen(gid, memory_item->mem_name, H5P_DEFAULT)) < 0) {
	    printf("ERROR in restore checkpoint for clamr/%s\n",memory_item->mem_name);
	  }
	  hid_t dtype = H5Dget_type(dset_id);
	  if( strcmp(memory_item->mem_name,"amesh_int_dist_vals") == 0 ) {

	    hsize_t stride[2], block[2];
	    hsize_t count[2], start[2];
	    hsize_t dims[2];
	    hid_t dataspace = H5Dget_space (dset_id);
	    H5Sget_simple_extent_dims(dataspace, dims, NULL );

	    start [0] = mype;
	    start [1] = 0;
	    count [0] = 1;
	    count [1] = dims[1];

	    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, NULL);

// 	    hid_t memspace = H5Screate_simple (2, count, NULL);  
// 	    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start, NULL, count, NULL);

	    
	    H5Dread( dset_id, dtype, H5S_ALL, dataspace, H5P_DEFAULT, mem_ptr);
	  }
	  H5Tclose(dtype);
	  H5Gclose(gid);
	  H5Dclose(dset_id);
	} else
	  printf("Error: unknown dataset %s\n",memory_item->mem_name);
      }
#endif
   }

}

void Crux::restore_begin(char *restart_file, int rollback_counter)
{
   rs_num = rollback_counter % num_of_rollback_states;

   cpu_timer_start(&trestore_time);

   if (restart_file != NULL){

      if (mype == 0) {
         printf("\n  ================================================================\n");
         printf(  "  Restoring state from disk file %s\n",restart_file);
         printf(  "  ================================================================\n\n");
      }

#ifdef HAVE_HDF5
      hid_t fapl_id;
#endif
      if(!USE_HDF5) {
#ifdef HAVE_MPI
	int iret = MPI_File_open(MPI_COMM_WORLD, restart_file, MPI_MODE_RDONLY | MPI_MODE_UNIQUE_OPEN, MPI_INFO_NULL, &mpi_restore_fp);
	if(iret != MPI_SUCCESS){
	  //printf("Could not write %s at iteration %d\n",restart_file,crux_int_vals[8]);
	  printf("Could not open restart file %s\n",restart_file);
	}

#else
	restore_fp = fopen(restart_file,"r");
	if(!restore_fp){
         //printf("Could not write %s at iteration %d\n",restart_file,crux_int_vals[8]);
	  printf("Could not open restart file %s\n",restart_file);
	}
#endif
      }
#ifdef HAVE_HDF5
      else {
	fapl_id = H5Pcreate( H5P_FILE_ACCESS );
	H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
	if( (h5_fid = H5Fopen(restart_file, H5F_ACC_RDONLY, fapl_id)) < 0)
	  printf("Could not open restart file %s\n",restart_file);
      }
      H5Pclose( fapl_id );
#endif
      restore_type = RESTORE_RESTART;
   } else if(crux_type == CRUX_IN_MEMORY){
      printf("Restoring state from memory rollback number %d rollback_counter %d\n",rs_num,rollback_counter);
      restore_fp = fmemopen(crux_data[rs_num], crux_data_size[rs_num], "r");
      restore_type = RESTORE_ROLLBACK;
   } else if(crux_type == CRUX_DISK){

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
#ifdef HAVE_MPI
   assert(name != NULL);
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, name, name_size, MPI_CHAR, &status);
# ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_CHAR, &count);
   printf("Read %d characters at line %d in file %s\n",count,__LINE__,__FILE__);
# endif

#else
  if(!USE_HDF5) {
   int name_read = fread(name,sizeof(char),name_size,restore_fp);
   if (name_read != name_size){
      printf("Warning: number of elements read %d is not equal to request %d\n",name_read,name_size);
   }

  }
# ifdef HAVE_HDF5
  else {
    
    // MSB
  }
# endif
#endif
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

#ifdef HAVE_HDF5
void Crux::restore_hdf5_values(void *vals, const char* dataset)
{
  hid_t dset_id;
  if( (dset_id = H5Dopen(h5_fid, dataset, H5P_DEFAULT)) < 0 )
    printf("Error: Cannot open dataset %s \n", dataset);

  hid_t dtype = H5Dget_type(dset_id);
  if( H5Dread(dset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals) < 0 )
    printf("Error: Cannot read dataset %s \n", dataset);

  H5Tclose(dtype);
  H5Dclose(dset_id);

}
#endif

int *Crux::restore_int_array(int *int_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(int_array != NULL);
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, int_array, (int)nelem, MPI_INT, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("Read %d integers at line %d in file %s\n",count,__LINE__,__FILE__);
#endif

#else
   size_t nelem_read = fread(int_array,sizeof(int),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
#endif
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
#ifdef HAVE_MPI
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, double_array, (int)nelem, MPI_DOUBLE, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE, &count);
   printf("Read %d doubles at line %d in file %s\n",count,__LINE__,__FILE__);
#endif
  
#else
   size_t nelem_read = fread(double_array,sizeof(double),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
#endif
   return(double_array);
}

int *Crux::restore_replicated_int_array(int *int_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(int_array != NULL);
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, int_array, (int)nelem, MPI_INT, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("Read %d integers at line %d in file %s\n",count,__LINE__,__FILE__);
#endif

#else
   size_t nelem_read = fread(int_array,sizeof(int),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
#endif
   return(int_array);
}

double *Crux::restore_replicated_double_array(double *double_array, size_t nelem)
{
#ifdef HAVE_MPI
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, double_array, (int)nelem, MPI_DOUBLE, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE, &count);
   printf("Read %d doubles at line %d in file %s\n",count,__LINE__,__FILE__);
#endif
  
#else
   size_t nelem_read = fread(double_array,sizeof(double),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
#endif
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

   if(!USE_HDF5) {
     fclose(restore_fp);
   }
#ifdef HAVE_HDF5
   else {
     H5Fclose( h5_fid );
   }
#endif     

}

int Crux::get_rollback_number()
{
  rollback_attempt++;
  return(checkpoint_counter % num_of_rollback_states);
}

/*
 * Helper function used to recursively print container contents
 * for container identified by "file_id" 
 * in read context identified by "rc_id"
 * and "my_rank" used to identify the process doing the reading / printing.
 */
#ifdef HDF5_FF
void
print_container_contents_ff( hid_t file_id, hid_t rc_id, int my_rank )
{
   herr_t ret;
   uint64_t cv;
   htri_t exists;
   char path_to_object[1024];
   hid_t obj_id;
   char name[3];
   int i;

   static int lvl = 0;     /* level in recursion - used to format printing */
   char preface[128];    
   char line[1024];
   char path[128];
   int attr_long_val[1];
   double attr_double_val[1];

   /* Get the container version for the read context */
   ret = H5RCget_version( rc_id, &cv ); 

   /* Set up the preface and adding version number */
   sprintf( preface, "r%d: cv %d: ", my_rank, (int)cv );

   /* Start the printing */
   if ( lvl == 0 ) {
      fprintf( stderr, "%s ----- Container Contents ------------\n", preface );
   } 

   /* Attributes */
   for ( i = 1; i < 3; i++ ) {
      if ( i == 1 ) {
	strcpy(path, "/mesh/");
	strcpy(name, "mesh_double_vals" );
      } else if (i == 2) {
	strcpy(path, "/state/");
	strcpy(name, "state_long_vals" );
      }
      ret = H5Aexists_by_name_ff( file_id, path, name, H5P_DEFAULT, &exists, rc_id, H5_EVENT_STACK_NULL ); 
      if ( exists ) { 
	hid_t attr_id;
	attr_id = H5Aopen_by_name_ff( file_id, path, name, H5P_DEFAULT, H5P_DEFAULT, rc_id, H5_EVENT_STACK_NULL ); 
	assert( attr_id >= 0 );
   
	 if(i == 1) {
	   ret = H5Aread_ff( attr_id,  H5T_NATIVE_DOUBLE, attr_double_val, rc_id, H5_EVENT_STACK_NULL ); 
	   fprintf( stderr, "%s %s%s   value: %f\n", preface, path, name, attr_double_val[0] );
	 } else if (i == 2) {
	   ret = H5Aread_ff( attr_id,  H5T_NATIVE_LONG, attr_long_val, rc_id, H5_EVENT_STACK_NULL ); 
	   fprintf( stderr, "%s %s%s   value: %ld\n", preface, path, name, attr_long_val[0] );
	 }

         ret = H5Aclose_ff( attr_id, H5_EVENT_STACK_NULL );
      }
   }

   hid_t dtype;
   /* Datasets */
   for ( i = 1; i < 7; i++ ) {
      if ( i == 1 ) { 
         strcpy( name, "i" );
         strcpy( path, "/mesh/" );
	 dtype = H5T_NATIVE_INT;
      } else if ( i == 2 ){ 
         strcpy( name, "j" );
         strcpy( path, "/mesh/" ); 
	 dtype = H5T_NATIVE_INT;
      } else if ( i == 3 ){
         strcpy( name, "level" );
         strcpy( path, "/mesh/" );
	 dtype = H5T_NATIVE_INT;
      } else if ( i == 4 ){
         strcpy( name, "H" );
         strcpy( path, "/state/" );
	 dtype = H5T_NATIVE_DOUBLE;
      } else if ( i == 5 ){
         strcpy( path, "/state/" );
         strcpy( name, "U" );
	 dtype = H5T_NATIVE_DOUBLE;
      } else if ( i == 6 ){
         strcpy( path, "/state/" );
         strcpy( name, "V" );
	 dtype = H5T_NATIVE_DOUBLE;
      }

      sprintf( path_to_object, "%s%s", path, name );
      ret = H5Lexists_ff( file_id, path_to_object, H5P_DEFAULT, &exists, rc_id, H5_EVENT_STACK_NULL );

      if ( exists ) { 
         hid_t dset_id;
         hid_t space_id;
         int nDims;
         hsize_t current_size[2];
         hsize_t totalSize;
         int *data_int;     
         double *data_double;              
         int k;

         dset_id = H5Dopen_ff( file_id, path_to_object, H5P_DEFAULT, rc_id, H5_EVENT_STACK_NULL ); 
         assert( dset_id >= 0 );

         space_id = H5Dget_space( dset_id ); assert ( space_id >= 0 );
         nDims = H5Sget_simple_extent_dims( space_id, current_size, NULL ); 
         assert( ( nDims == 1 ) || ( nDims == 2 ) );

         if ( nDims == 1 ) {
            totalSize = current_size[0];
         } else {
            totalSize = current_size[0] * current_size[1];
         }

	 printf("r%d: object : %s \n", my_rank, path_to_object);
	 if (i < 4) {
	   data_int = (int *)calloc( totalSize, sizeof(int) ); assert( data_int != NULL );
	   ret = H5Dread_ff( dset_id, dtype, space_id, space_id, H5P_DEFAULT, data_int, rc_id, H5_EVENT_STACK_NULL );
	   if ( nDims == 1 ) {
	     for ( k = 0; k < totalSize; k++ ) {
               if(verbose_io) printf( "%d ", data_int[k] );
	     }
	   } else {
	     int r, c;
	     k = 0;
	     for ( r = 0; r < current_size[0]; r++ ) {
               for ( c = 0; c < current_size[1]; c++ ) {
		 if(verbose_io) printf( "%d ", data_int[k] );
		 k++;
               }
	     }
	   }
	   free( data_int );
	 } else {
	   data_double = (double *)calloc( totalSize, sizeof(double) ); assert( data_double != NULL );
	   ret = H5Dread_ff( dset_id, dtype, space_id, space_id, H5P_DEFAULT, data_double, rc_id, H5_EVENT_STACK_NULL );
	   if ( nDims == 1 ) {
	     for ( k = 0; k < totalSize; k++ ) {
               if(verbose_io) printf( "%f ", data_double[k] );
	     }
	   } else {
	     int r, c;
	     k = 0;
	     for ( r = 0; r < current_size[0]; r++ ) {
               for ( c = 0; c < current_size[1]; c++ ) {
		 if(verbose_io) printf( "%f ", data_double[k] );
		 k++;
               }
	     }
	   }
	   free( data_double );
	 }
         ret = H5Dclose_ff( dset_id, H5_EVENT_STACK_NULL ); 
         ret = H5Sclose( space_id ); 
      } 
   }

   /* End printing */
   if ( lvl == 0 ) {
      fprintf( stderr, "%s -----------------\n", preface );
   }

   return;
}
#else
#  if HAVE_HDF5 
void
print_container_contents( hid_t file_id, int my_rank )
{
   herr_t ret;
   uint64_t cv;
   htri_t exists;
   char path_to_object[1024];
   hid_t obj_id;
   char name[3];
   int i;

   static int lvl = 0;     /* level in recursion - used to format printing */
   char preface[128];    
   char line[1024];
   char path[128];
   int attr_long_val[1];
   double attr_double_val[1];

   /* Set up the preface and adding version number */
   sprintf( preface, "r%d: : ", my_rank );

   /* Start the printing */
   if ( lvl == 0 ) {
      fprintf( stderr, "%s ----- Container Contents ------------\n", preface );
   } 

   /* Attributes */
   for ( i = 1; i < 3; i++ ) {
      if ( i == 1 ) {
	strcpy(path, "/mesh/");
	strcpy(name, "mesh_double_vals" );
      } else if (i == 2) {
	strcpy(path, "/state/");
	strcpy(name, "state_long_vals" );
      } 
      if (  H5Aexists_by_name( file_id, path, name, H5P_DEFAULT) != 0 ) { 
	hid_t attr_id;
	attr_id = H5Aopen_by_name( file_id, path, name, H5P_DEFAULT, H5P_DEFAULT); 
	assert( attr_id >= 0 );
   
	 if(i == 1) {
	   ret = H5Aread( attr_id,  H5T_NATIVE_DOUBLE, attr_double_val); 
	   fprintf( stderr, "%s %s%s   value: %f\n", preface, path, name, attr_double_val[0] );
	 } else if (i == 2) {
	   ret = H5Aread( attr_id,  H5T_NATIVE_LONG, attr_long_val); 
	   fprintf( stderr, "%s %s%s   value: %ld\n", preface, path, name, attr_long_val[0] );
	 }

         ret = H5Aclose( attr_id); 
      }
   }

   hid_t dtype;
   /* Datasets */
   for ( i = 1; i < 7; i++ ) {
      if ( i == 1 ) { 
         strcpy( name, "i" );
         strcpy( path, "/mesh/" );
	 dtype = H5T_NATIVE_INT;
      } else if ( i == 2 ){ 
         strcpy( name, "j" );
         strcpy( path, "/mesh/" ); 
	 dtype = H5T_NATIVE_INT;
      } else if ( i == 3 ){
         strcpy( name, "level" );
         strcpy( path, "/mesh/" );
	 dtype = H5T_NATIVE_INT;
      } else if ( i == 4 ){
         strcpy( name, "H" );
         strcpy( path, "/state/" );
	 dtype = H5T_NATIVE_DOUBLE;
      } else if ( i == 5 ){
         strcpy( path, "/state/" );
         strcpy( name, "U" );
	 dtype = H5T_NATIVE_DOUBLE;
      } else if ( i == 6 ){
         strcpy( path, "/state/" );
         strcpy( name, "V" );
	 dtype = H5T_NATIVE_DOUBLE;
      }

      sprintf( path_to_object, "%s%s", path, name );
 
      if ( H5Lexists( file_id, path_to_object, H5P_DEFAULT) != 0 ) {

	hid_t dset_id;
	hid_t space_id;
	int nDims;
	hsize_t current_size[2];
	hsize_t totalSize;
	int *data_int;     
	double *data_double;              
	int k;

         dset_id = H5Dopen( file_id, path_to_object, H5P_DEFAULT ); 
         assert( dset_id >= 0 );

         space_id = H5Dget_space( dset_id ); assert ( space_id >= 0 );
         nDims = H5Sget_simple_extent_dims( space_id, current_size, NULL ); 
         assert( ( nDims == 1 ) || ( nDims == 2 ) );

         if ( nDims == 1 ) {
            totalSize = current_size[0];
         } else {
            totalSize = current_size[0] * current_size[1];
         }

	 printf("r%d: object : %s \n", my_rank, path_to_object);
	 if (i < 4) {
	   data_int = (int *)calloc( totalSize, sizeof(int) ); assert( data_int != NULL );
	   ret = H5Dread( dset_id, dtype, space_id, space_id, H5P_DEFAULT, data_int);
 	   if ( nDims == 1 ) {
 	     for ( k = 0; k < totalSize; k++ ) {
	       if(verbose_io) printf( "%d ", data_int[k] );
	       //   if(verbose_io) printf( "%d ", data_int[k] );
 	     }
	   } else {
	     int r, c;
	     k = 0;
	     for ( r = 0; r < current_size[0]; r++ ) {
               for ( c = 0; c < current_size[1]; c++ ) {
		 if(verbose_io) printf( "%d ", data_int[k] );
		 //	 if(verbose_io) printf( "%d ", data_int[k] );
		 k++;
               }
	     }
	   free( data_int );
	   }
	 } else {
	   data_double = (double *)calloc( totalSize, sizeof(double) ); assert( data_double != NULL );
	   ret = H5Dread( dset_id, dtype, space_id, space_id, H5P_DEFAULT, data_double);
	   if ( nDims == 1 ) {
	     for ( k = 0; k < totalSize; k++ ) {
               if(verbose_io) printf( "%f ", data_double[k] );
	       // MSB  if(verbose_io) printf( "%f ", data_double[k] );
	     }
	   } else {
	     int r, c;
	     k = 0;
	     for ( r = 0; r < current_size[0]; r++ ) {
               for ( c = 0; c < current_size[1]; c++ ) {
		 if(verbose_io) printf( "%f ", data_double[k] ); // MSB uncomm
		 k++;
               }
	     }
	   }
	   free( data_double );
	 }
    
         ret = H5Dclose( dset_id );
         ret = H5Sclose( space_id ); 
      } 
   }

   /* End printing */
   if ( lvl == 0 ) {
      fprintf( stderr, "%s -----------------\n", preface );
   }

   return;
}
#endif
#endif
void Crux::set_crux_type(int crux_type_in)
{
  crux_type = crux_type_in;
}
