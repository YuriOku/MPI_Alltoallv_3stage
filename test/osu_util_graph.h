
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

#include <stdio.h>
#include <stdlib.h>

#define OMB_GRAPH_FLOATING_PRECISION 15
#define OMB_DUMB_COL_SIZE            120
#define OMB_DUMB_ROW_SIZE            40
#define OMB_PNG_COL_SIZE             2048
#define OMB_PNG_ROW_SIZE             1080
#define OMB_ENV_ROW_SIZE             "OMB_ROW_SIZE"
#define OMB_ENV_COL_SIZE             "OMB_COL_SIZE"
#define OMB_ENV_SLURM_ROW_SIZE       "SLURM_PTY_WIN_ROW"
#define OMB_ENV_SLURM_COL_SIZE       "SLURM_PTY_WIN_COL"
#define OMB_PNG_MAX_FILE_LENGTH      30
#define OMB_CMD_MAX_LENGTH           1024
#define OMB_GNUPLOT_PNG_FONT_SIZE    20
#define OMB_GNUPLOT_DATA_FILENAME    "plot_data.dat"
#define OMB_3D_GRAPH_PARTS           2 /*Not to be modified*/

#ifndef _GNUPLOT_BUILD_PATH_
#define OMB_GNUPLOT_PATH "gnuplot"
#else
#define OMB_GNUPLOT_PATH _GNUPLOT_BUILD_PATH_
#endif
#ifndef _CONVERT_BUILD_PATH_
#define OMB_CONVERT_PATH "convert"
#else
#define OMB_CONVERT_PATH _CONVERT_BUILD_PATH_
#endif

typedef struct omb_terminal_size {
    size_t row_size;
    size_t col_size;
} omb_terminal_size_t;

typedef struct omb_graph_data {
    double *data;
    size_t length;
    double avg;
    size_t message_size;
} omb_graph_data_t;

typedef struct omb_graph_options {
    FILE *gnuplot_pointer;
    size_t number_of_graphs;
    omb_graph_data_t **graph_datas;
} omb_graph_options_t;

struct omb_terminal_size omb_get_terminal_size();
int omb_graph_init(omb_graph_options_t *graph_options);
void omb_graph_options_init(omb_graph_options_t *graph_options);
void omb_graph_plot(omb_graph_options_t *graph_options, const char *filename);
void omb_graph_combined_plot(omb_graph_options_t *graph_options,
                             const char *filename);
void omb_graph_allocate_data_buffer(omb_graph_options_t *graph_options,
                                    size_t message_size, size_t length);
int omb_graph_free_data_buffers(omb_graph_options_t *graph_options);
void omb_graph_close(omb_graph_options_t *graph_options);
void omb_graph_allocate_and_get_data_buffer(omb_graph_data_t **graph_data,
                                            omb_graph_options_t *graph_options,
                                            size_t message_size, size_t length);
