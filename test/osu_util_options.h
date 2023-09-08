
/*                          COPYRIGHT
*
* Copyright (c) 2001-2023, The Ohio State University. All rights
* reserved.
*
* The OMB (OSU Micro Benchmarks) software package is developed by the team
* members of The Ohio State University's Network-Based Computing Laboratory
* (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
*
* Contact:
* Prof. Dhabaleswar K. (DK) Panda
* Dept. of Computer Science and Engineering
* The Ohio State University
* 2015 Neil Avenue
* Columbus, OH - 43210-1277
* Tel: (614)-292-5199; Fax: (614)-292-2911
* E-mail:panda@cse.ohio-state.edu
*
* This program is available under BSD licensing.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are
* met:
*
* (1) Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
*
* (2) Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
*
* (3) Neither the name of The Ohio State University nor the names of
* their contributors may be used to endorse or promote products derived
* from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
* OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
* LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
* THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef OMB_UTIL_OP_H
#define OMB_UTIL_OP_H                 1
#define OMBOP_GET_NONACCEL_NAME(A, B) OMBOP__##A##__##B
#define OMBOP_GET_ACCEL_NAME(A, B)    OMBOP__ACCEL__##A##__##B
#define OMBOP_OPTSTR_BLK(bench, subtype)                                       \
    if (accel_enabled) {                                                       \
        optstring = OMBOP_GET_ACCEL_NAME(bench, subtype);                      \
    } else {                                                                   \
        optstring = OMBOP_GET_NONACCEL_NAME(bench, subtype);                   \
    }
#define OMBOP_OPTSTR_CUDA_BLK(bench, subtype)                                  \
    if (accel_enabled) {                                                       \
        optstring = (CUDA_KERNEL_ENABLED) ?                                    \
                        OMBOP_GET_ACCEL_NAME(bench, subtype) "r:" :            \
                        OMBOP_GET_ACCEL_NAME(bench, subtype);                  \
    } else {                                                                   \
        optstring = OMBOP_GET_NONACCEL_NAME(bench, subtype);                   \
    }
#define OMBOP_LONG_OPTIONS_ALL                                                 \
    {                                                                          \
        {"help", no_argument, 0, 'h'},                                         \
        {"version", no_argument, 0, 'v'},                                      \
        {"full", no_argument, 0, 'f'},                                         \
        {"message-size", required_argument, 0, 'm'},                           \
        {"window-size", required_argument, 0, 'W'},                            \
        {"num-test-calls", required_argument, 0, 't'},                         \
        {"iterations", required_argument, 0, 'i'},                             \
        {"warmup", required_argument, 0, 'x'},                                 \
        {"array-size", required_argument, 0, 'a'},                             \
        {"sync-option", required_argument, 0, 's'},                            \
        {"win-options", required_argument, 0, 'w'},                            \
        {"mem-limit", required_argument, 0, 'M'},                              \
        {"accelerator", required_argument, 0, 'd'},                            \
        {"cuda-target", required_argument, 0, 'r'},                            \
        {"print-rate", required_argument, 0, 'R'},                             \
        {"num-pairs", required_argument, 0, 'p'},                              \
        {"vary-window", required_argument, 0, 'V'},                            \
        {"validation", no_argument, 0, 'c'},                                   \
        {"buffer-num", required_argument, 0, 'b'},                             \
        {"validation-warmup", required_argument, 0, 'u'},                      \
        {"graph", required_argument, 0, 'G'},                                  \
        {"papi", required_argument, 0, 'P'},                                   \
        {"ddt", required_argument, 0, 'D'},                                    \
        {"nhbr", required_argument, 0, 'N'},                                   \
        {"type", required_argument, 0, 'T'},                                   \
        {"session", no_argument, 0, 'I'},                                      \
        {"in-place", no_argument, 0, 'l'},                                     \
        {"root-rank", required_argument, 0, 'k'}                               \
    }
/*OMBOP[__ACCEL]__<options.bench>__<options.subtype>*/
#define OMBOP__PT2PT__LAT                     "+:hvm:x:i:b:cu:G:D:P:T:I"
#define OMBOP__ACCEL__PT2PT__LAT              "+:x:i:m:d:hvcu:G:D:T:I"
#define OMBOP__PT2PT__BW                      "+:hvm:x:i:t:W:b:cu:G:D:P:T:I"
#define OMBOP__ACCEL__PT2PT__BW               "+:x:i:t:m:d:W:hvb:cu:G:D:T:I"
#define OMBOP__PT2PT__LAT_MT                  "+:hvm:x:i:t:cu:G:D:T:I"
#define OMBOP__ACCEL__PT2PT__LAT_MT           OMBOP__ACCEL__PT2PT__LAT
#define OMBOP__PT2PT__LAT_MP                  "+:hvm:x:i:t:cu:G:D:P:T:I"
#define OMBOP__ACCEL__PT2PT__LAT_MP           OMBOP__ACCEL__PT2PT__LAT
#define OMBOP__COLLECTIVE__ALLTOALL           "+:hvfm:i:x:M:a:cu:G:D:P:T:IlE:"
#define OMBOP__ACCEL__COLLECTIVE__ALLTOALL    "+:d:hvfm:i:x:M:a:cu:G:D:T:IlE:"
#define OMBOP__COLLECTIVE__GATHER             OMBOP__COLLECTIVE__ALLTOALL "k:"
#define OMBOP__ACCEL__COLLECTIVE__GATHER      OMBOP__ACCEL__COLLECTIVE__ALLTOALL "k:"
#define OMBOP__COLLECTIVE__ALL_GATHER         OMBOP__COLLECTIVE__ALLTOALL
#define OMBOP__ACCEL__COLLECTIVE__ALL_GATHER  OMBOP__ACCEL__COLLECTIVE__ALLTOALL
#define OMBOP__COLLECTIVE__SCATTER            OMBOP__COLLECTIVE__ALLTOALL "k:"
#define OMBOP__ACCEL__COLLECTIVE__SCATTER     OMBOP__ACCEL__COLLECTIVE__ALLTOALL "k:"
#define OMBOP__COLLECTIVE__BCAST              "+:hvfm:i:x:M:a:cu:G:D:P:T:I"
#define OMBOP__ACCEL__COLLECTIVE__BCAST       "+:d:hvfm:i:x:M:a:cu:G:D:T:I"
#define OMBOP__COLLECTIVE__NHBR_GATHER        "+:hvfm:i:x:M:a:cu:N:G:D:P:T:I"
#define OMBOP__ACCEL__COLLECTIVE__NHBR_GATHER "+:hvfm:i:x:M:a:cu:N:G:D:T:I"
#define OMBOP__COLLECTIVE__NHBR_ALLTOALL      OMBOP__COLLECTIVE__NHBR_GATHER
#define OMBOP__ACCEL__COLLECTIVE__NHBR_ALLTOALL                                \
    OMBOP__ACCEL__COLLECTIVE__NHBR_GATHER
#define OMBOP__COLLECTIVE__BARRIER               "+:hvfm:i:x:M:a:u:G:P:I"
#define OMBOP__ACCEL__COLLECTIVE__BARRIER        "+:d:hvfm:i:x:M:a:u:G:I"
#define OMBOP__COLLECTIVE__LAT                   "+:hvfm:i:x:M:a:"
#define OMBOP__ACCEL__COLLECTIVE__LAT            "+:d:hvfm:i:x:M:a:"
#define OMBOP__COLLECTIVE__ALL_REDUCE            "+:hvfm:i:x:M:a:cu:G:P:T:Il"
#define OMBOP__ACCEL__COLLECTIVE__ALL_REDUCE     "+:d:hvfm:i:x:M:a:cu:G:T:Il"
#define OMBOP__COLLECTIVE__REDUCE                OMBOP__COLLECTIVE__ALL_REDUCE "k:"
#define OMBOP__ACCEL__COLLECTIVE__REDUCE         OMBOP__ACCEL__COLLECTIVE__ALL_REDUCE "k:"
#define OMBOP__COLLECTIVE__REDUCE_SCATTER        OMBOP__COLLECTIVE__ALL_REDUCE
#define OMBOP__ACCEL__COLLECTIVE__REDUCE_SCATTER OMBOP__ACCEL__COLLECTIVE__ALL_REDUCE
#define OMBOP__COLLECTIVE__NBC_BARRIER           "+:hvfm:i:x:M:t:a:G:P:I"
#define OMBOP__ACCEL__COLLECTIVE__NBC_BARRIER    "+:d:hvfm:i:x:M:t:a:G:I"
#define OMBOP__COLLECTIVE__NBC_ALLTOALL          "+:hvfm:i:x:M:t:a:cu:G:D:P:T:Il"
#define OMBOP__ACCEL__COLLECTIVE__NBC_ALLTOALL   "+:d:hvfm:i:x:M:t:a:cu:G:D:T:Il"
#define OMBOP__COLLECTIVE__NBC_GATHER            OMBOP__COLLECTIVE__NBC_ALLTOALL "k:"
#define OMBOP__ACCEL__COLLECTIVE__NBC_GATHER     OMBOP__ACCEL__COLLECTIVE__NBC_ALLTOALL "k:"
#define OMBOP__COLLECTIVE__NBC_ALL_GATHER        OMBOP__COLLECTIVE__NBC_ALLTOALL
#define OMBOP__ACCEL__COLLECTIVE__NBC_ALL_GATHER OMBOP__ACCEL__COLLECTIVE__NBC_ALLTOALL
#define OMBOP__COLLECTIVE__NBC_SCATTER           OMBOP__COLLECTIVE__NBC_ALLTOALL "k:"
#define OMBOP__ACCEL__COLLECTIVE__NBC_SCATTER    OMBOP__ACCEL__COLLECTIVE__NBC_ALLTOALL "k:"
#define OMBOP__COLLECTIVE__NBC_BCAST          "+:hvfm:i:x:M:t:a:cu:G:D:P:T:I"
#define OMBOP__ACCEL__COLLECTIVE__NBC_BCAST   "+:d:hvfm:i:x:M:t:a:cu:G:D:T:I"
#define OMBOP__COLLECTIVE__NBC_ALL_REDUCE         "+:hvfm:i:x:M:t:a:cu:G:P:T:Il"
#define OMBOP__ACCEL__COLLECTIVE__NBC_ALL_REDUCE  "+:d:hvfm:i:x:M:t:a:cu:G:T:Il"
#define OMBOP__COLLECTIVE__NBC_REDUCE             OMBOP__COLLECTIVE__NBC_ALL_REDUCE "k:"
#define OMBOP__ACCEL__COLLECTIVE__NBC_REDUCE      OMBOP__ACCEL__COLLECTIVE__NBC_ALL_REDUCE "k:"
#define OMBOP__COLLECTIVE__NBC_REDUCE_SCATTER     OMBOP__COLLECTIVE__NBC_ALL_REDUCE
#define OMBOP__ACCEL__COLLECTIVE__NBC_REDUCE_SCATTER                           \
    OMBOP__ACCEL__COLLECTIVE__NBC_ALL_REDUCE
#define OMBOP__COLLECTIVE__NBC_NHBR_GATHER        "+:hvfm:i:x:M:t:a:cu:N:G:D:P:T:I"
#define OMBOP__ACCEL__COLLECTIVE__NBC_NHBR_GATHER "+:hvfm:i:x:M:t:a:cu:N:G:D:T:I"
#define OMBOP__COLLECTIVE__NBC_NHBR_ALLTOALL      OMBOP__COLLECTIVE__NBC_NHBR_GATHER
#define OMBOP__ACCEL__COLLECTIVE__NBC_NHBR_ALLTOALL                            \
    OMBOP__ACCEL__COLLECTIVE__NBC_NHBR_GATHER
#define OMBOP__ONE_SIDED__BW         "+:w:s:hvm:x:i:W:G:P:I"
#define OMBOP__ACCEL__ONE_SIDED__BW  "+:w:s:hvm:d:x:i:W:G:I"
#define OMBOP__ONE_SIDED__LAT        "+:w:s:hvm:x:i:G:P:I"
#define OMBOP__ACCEL__ONE_SIDED__LAT "+:w:s:hvm:d:x:i:G:I"
#define OMBOP__MBW_MR                "p:W:R:x:i:m:Vhvb:cu:G:D:P:T:I"
#define OMBOP__ACCEL__MBW_MR         "p:W:R:x:i:m:d:Vhvb:cu:G:D:T:I"
#define OMBOP__OSHM                  ":hvfm:i:M:";
#define OMBOP__UPC                   OMBOP__OSHM
#define OMBOP__UPCXX                 OMBOP__OSHM
#define OMBOP__STARTUP__INIT         "I"
#endif
