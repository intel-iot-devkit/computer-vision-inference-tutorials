// Copyright (c) 2016 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly


#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifdef cl_intel_simd_operations_placeholder
#define cl_intel_subgroups
#define sub_group_broadcast             intel_simd_shuffle
#endif

#define ENABLE_KERNEL_ALL

#if defined(ENABLE_KERNEL_ALL) || defined(ENABLE_KERNEL_UNOPTIMIZED)

// Simple unoptimized version of the algorithm:
// Assumes square matrices (M = K = N).
// Note: A and B are source matrices:
//  A is M rows by K columns
//  B is K rows by N columns
// C is the destination matrix:
//  C is M rows by N columns
__kernel void Unoptimized(__global const float *src0,
                      __global const float *src1,
                      __global float *dst,
                      int width0,
                      int width1)
{
    const int N = width1;
    const int K = width0;    // since matrices are square

    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float sum = 0.0f;
    for( int i = 0; i < K; i++ )
    {
        sum += 
            src0[ i + y * K ] *   // walking across A = src0
            src1[ x + i * N ];    // walking down B = src1
    }
    dst[ x + y * N ] = sum;
}

#endif






////////////////////////////////////////////////////////////////
// L3_SIMD_4x8x8

#if defined(cl_intel_subgroups) && ( defined(ENABLE_KERNEL_ALL) || defined(ENABLE_KERNEL_L3_SIMD_4x8x8) )

#define VEC_SIZE        4
#define LWG_HEIGHT      8
#define TILE_M          8
#define TILE_K          32
#define TILE_N          32

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void L3_SIMD_4x8x8(
    const __global float4 *src0,
    const __global float4 *src1,
    __global float4 *dst,
    int width0,
    int width1)
{
    width0 /= VEC_SIZE;
    width1 /= VEC_SIZE;

    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);

    // Result ctile is M rows x N columns
    // M = 8, we have 1 rows of work-items, so we need 8/1 = 8 results down
    // N = 32, we have 8 columns of work-items, so we need 32/8 = 4 results across = 1 float4s across

    float4 dot00 = (float4)(0.f);
    float4 dot01 = (float4)(0.f);
    float4 dot02 = (float4)(0.f);
    float4 dot03 = (float4)(0.f);
    float4 dot04 = (float4)(0.f);
    float4 dot05 = (float4)(0.f);
    float4 dot06 = (float4)(0.f);
    float4 dot07 = (float4)(0.f);

    __global float4 *dst_write0 = dst + local_x + ( group_x * ( TILE_N / VEC_SIZE ) ) + ( group_y * LWG_HEIGHT * TILE_M + 8 * local_y ) * width1;

    // Src0 is used directly as atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    // M = 8, we have 1 rows of work-items, so we need 8/1 = 8 rows.
    // K = 32, we have 8 columns of work-items, so we need 32/8 = 4 floats across = 1 float4s across
    const __global float4 *src0_read = src0 + local_x + ( group_y * LWG_HEIGHT * TILE_M + 8 * local_y ) * width0;

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    // K = 32, we'll process four rows at a time
    // N = 32, we have 8 columns of work-items, so we need 32/8 = 4 floats across = 1 float4s across
    const __global float4 *src1_read0 = src1 + local_x + ( group_x * ( TILE_N / VEC_SIZE ) );

    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do
    {
        const float4 arow0 = src0_read[ 0 * width0 ];
        const float4 arow1 = src0_read[ 1 * width0 ];
        const float4 arow2 = src0_read[ 2 * width0 ];
        const float4 arow3 = src0_read[ 3 * width0 ];
        const float4 arow4 = src0_read[ 4 * width0 ];
        const float4 arow5 = src0_read[ 5 * width0 ];
        const float4 arow6 = src0_read[ 6 * width0 ];
        const float4 arow7 = src0_read[ 7 * width0 ];

#define ITERATION( _index ) \
        {   \
            const float4 a0 = intel_sub_group_shuffle( arow0, _index ); \
            const float4 a1 = intel_sub_group_shuffle( arow1, _index ); \
            const float4 a2 = intel_sub_group_shuffle( arow2, _index ); \
            const float4 a3 = intel_sub_group_shuffle( arow3, _index ); \
            const float4 a4 = intel_sub_group_shuffle( arow4, _index ); \
            const float4 a5 = intel_sub_group_shuffle( arow5, _index ); \
            const float4 a6 = intel_sub_group_shuffle( arow6, _index ); \
            const float4 a7 = intel_sub_group_shuffle( arow7, _index ); \
            const float4 brow00 = src1_read0[ 0 ];   src1_read0 += width1;    \
            const float4 brow01 = src1_read0[ 0 ];   src1_read0 += width1;    \
            const float4 brow02 = src1_read0[ 0 ];   src1_read0 += width1;    \
            const float4 brow03 = src1_read0[ 0 ];   src1_read0 += width1;    \
            dot00 = mad(brow00, (float4) a0.x, dot00);  \
            dot00 = mad(brow01, (float4) a0.y, dot00);  \
            dot00 = mad(brow02, (float4) a0.z, dot00);  \
            dot00 = mad(brow03, (float4) a0.w, dot00);  \
            dot01 = mad(brow00, (float4) a1.x, dot01);  \
            dot01 = mad(brow01, (float4) a1.y, dot01);  \
            dot01 = mad(brow02, (float4) a1.z, dot01);  \
            dot01 = mad(brow03, (float4) a1.w, dot01);  \
            dot02 = mad(brow00, (float4) a2.x, dot02);  \
            dot02 = mad(brow01, (float4) a2.y, dot02);  \
            dot02 = mad(brow02, (float4) a2.z, dot02);  \
            dot02 = mad(brow03, (float4) a2.w, dot02);  \
            dot03 = mad(brow00, (float4) a3.x, dot03);  \
            dot03 = mad(brow01, (float4) a3.y, dot03);  \
            dot03 = mad(brow02, (float4) a3.z, dot03);  \
            dot03 = mad(brow03, (float4) a3.w, dot03);  \
            dot04 = mad(brow00, (float4) a4.x, dot04);  \
            dot04 = mad(brow01, (float4) a4.y, dot04);  \
            dot04 = mad(brow02, (float4) a4.z, dot04);  \
            dot04 = mad(brow03, (float4) a4.w, dot04);  \
            dot05 = mad(brow00, (float4) a5.x, dot05);  \
            dot05 = mad(brow01, (float4) a5.y, dot05);  \
            dot05 = mad(brow02, (float4) a5.z, dot05);  \
            dot05 = mad(brow03, (float4) a5.w, dot05);  \
            dot06 = mad(brow00, (float4) a6.x, dot06);  \
            dot06 = mad(brow01, (float4) a6.y, dot06);  \
            dot06 = mad(brow02, (float4) a6.z, dot06);  \
            dot06 = mad(brow03, (float4) a6.w, dot06);  \
            dot07 = mad(brow00, (float4) a7.x, dot07);  \
            dot07 = mad(brow01, (float4) a7.y, dot07);  \
            dot07 = mad(brow02, (float4) a7.z, dot07);  \
            dot07 = mad(brow03, (float4) a7.w, dot07);  \
        }

        // If I had #pragma unroll I wouldn't need to do this manually...

        // We need K/VEC_SIZE iterations.
        // K = 32, VEC_SIZE = 4
        // So, 32/4 = 8 iterations.
        ITERATION( 0 );
        ITERATION( 1 );
        ITERATION( 2 );
        ITERATION( 3 );
        ITERATION( 4 );
        ITERATION( 5 );
        ITERATION( 6 );
        ITERATION( 7 );

#undef ITERATION

        src0_read += TILE_K / VEC_SIZE;
        w += TILE_K / VEC_SIZE;
    }
    while( w < width0 );

    dst_write0[ 0 ] = dot00;  dst_write0 += width1;
    dst_write0[ 0 ] = dot01;  dst_write0 += width1;
    dst_write0[ 0 ] = dot02;  dst_write0 += width1;
    dst_write0[ 0 ] = dot03;  dst_write0 += width1;
    dst_write0[ 0 ] = dot04;  dst_write0 += width1;
    dst_write0[ 0 ] = dot05;  dst_write0 += width1;
    dst_write0[ 0 ] = dot06;  dst_write0 += width1;
    dst_write0[ 0 ] = dot07;  dst_write0 += width1;
}

#undef VEC_SIZE
#undef LWG_HEIGHT
#undef TILE_M
#undef TILE_K
#undef TILE_N

#endif







////////////////////////////////////////////////////////////////
// MediaBlockRW_SIMD_2x32

#if defined(cl_intel_subgroups) && ( defined(ENABLE_KERNEL_ALL) || defined(ENABLE_KERNEL_MEDIABLOCKREADWRITE_SIMD_2x32) )

#define TILE_M          32
#define TILE_K          8
#define TILE_N          16

__attribute__((reqd_work_group_size(8, 1, 1)))
__kernel void MediaBlockRW_SIMD_2x32(
    __read_only image2d_t src0,
    __read_only image2d_t src1,
    __write_only image2d_t dst,
    int width0,
    int width1)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);

    // Result ctile is M rows x N columns
    // M = 32, we have 1 rows of work-items, so we need 32/1 = 32 results down = 4 x float8
    // N = 16, we have 8 columns of work-items, so we need 16/8 = 2 results across
    // Note: ctile is 64 registers

    float8 blockC00 = 0.0f;
    float8 blockC01 = 0.0f;
    float8 blockC02 = 0.0f;
    float8 blockC03 = 0.0f;
    float8 blockC10 = 0.0f;
    float8 blockC11 = 0.0f;
    float8 blockC12 = 0.0f;
    float8 blockC13 = 0.0f;

    // Src0 is directly used as atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int2    coordA = (int2)( 0, group_y * TILE_M );

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    int2    coordB = (int2)( ( group_x * TILE_N ) * sizeof(uint), 0 );

    // Walk ACROSS src0 and DOWN src1:
    do
    {
#define TRANSPOSE_BLOCK_8( _block, _col )   \
        (float8)( sub_group_broadcast( _block.s0, _col ),   \
                  sub_group_broadcast( _block.s1, _col ),   \
                  sub_group_broadcast( _block.s2, _col ),   \
                  sub_group_broadcast( _block.s3, _col ),   \
                  sub_group_broadcast( _block.s4, _col ),   \
                  sub_group_broadcast( _block.s5, _col ),   \
                  sub_group_broadcast( _block.s6, _col ),   \
                  sub_group_broadcast( _block.s7, _col ) );

#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )    \
        {   \
            const float8    acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );    \
            const float8    acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );    \
            const float8    acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );    \
            const float8    acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );    \
            const float8    acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );    \
            const float8    acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );    \
            const float8    acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );    \
            const float8    acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );    \
            _result = mad( (float8)(_blockB.s0), acol0, _result );      \
            _result = mad( (float8)(_blockB.s1), acol1, _result );      \
            _result = mad( (float8)(_blockB.s2), acol2, _result );      \
            _result = mad( (float8)(_blockB.s3), acol3, _result );      \
            _result = mad( (float8)(_blockB.s4), acol4, _result );      \
            _result = mad( (float8)(_blockB.s5), acol5, _result );      \
            _result = mad( (float8)(_blockB.s6), acol6, _result );      \
            _result = mad( (float8)(_blockB.s7), acol7, _result );      \
        }

        // atile is M rows x K columns
        // M = 32, we have 1 row of work-items, so each work-item must load 32/1 = 32 rows
        // K = 8, we have 8 columns of work-items, so each work-item must load 8/8 = 1 column
        // Note: atile is up to 32 registers

        // btile is K rows x N columns
        // K = 8, we have 1 row of work-items, so each work-item must load 8/1 = 8 rows
        // N = 16, we have 8 columns of work-items, so each work-item must load 16/8 = 2 column
        // Note: btile is up to 16 registers

        // It's annoying that we have to do this interleaving manually -
        // at some point we should investigate why the region prescheduler
        // isn't doing this for us.

        int2    coordBTemp = coordB;
        float8  blockB00 = as_float8( intel_sub_group_block_read8( src1, coordBTemp ) );    coordBTemp.x += 8 * sizeof(uint);
        float8  blockB10 = as_float8( intel_sub_group_block_read8( src1, coordBTemp ) );    coordB.y += TILE_K;

        int2    coordATemp = coordA;
        float8  blockA00 = as_float8( intel_sub_group_block_read8( src0, coordATemp ) );    coordATemp.y += 8;
        MULTIPLY_BLOCKS_8x8( blockC00, blockA00, blockB00 );
        MULTIPLY_BLOCKS_8x8( blockC10, blockA00, blockB10 );

        float8  blockA01 = as_float8( intel_sub_group_block_read8( src0, coordATemp ) );    coordATemp.y += 8;
        MULTIPLY_BLOCKS_8x8( blockC01, blockA01, blockB00 );
        MULTIPLY_BLOCKS_8x8( blockC11, blockA01, blockB10 );

        float8  blockA02 = as_float8( intel_sub_group_block_read8( src0, coordATemp ) );    coordATemp.y += 8;
        MULTIPLY_BLOCKS_8x8( blockC02, blockA02, blockB00 );
        MULTIPLY_BLOCKS_8x8( blockC12, blockA02, blockB10 );

        float8  blockA03 = as_float8( intel_sub_group_block_read8( src0, coordATemp ) );    coordA.x += TILE_K * sizeof(uint);
        MULTIPLY_BLOCKS_8x8( blockC03, blockA03, blockB00 );
        MULTIPLY_BLOCKS_8x8( blockC13, blockA03, blockB10 );

#undef TRANSPOSE_BLOCK_8
#undef MULTIPLY_BLOCKS_8x8
    }
    while( coordB.y < width0 );

    int2    coordDst = (int2)( ( group_x * TILE_N ) * sizeof(uint), ( group_y * TILE_M ) );

    int2    coordDstTemp = coordDst;
    intel_sub_group_block_write8( dst, coordDstTemp, as_uint8( blockC00 ) );    coordDstTemp.y += 8;
    intel_sub_group_block_write8( dst, coordDstTemp, as_uint8( blockC01 ) );    coordDstTemp.y += 8;
    intel_sub_group_block_write8( dst, coordDstTemp, as_uint8( blockC02 ) );    coordDstTemp.y += 8;
    intel_sub_group_block_write8( dst, coordDstTemp, as_uint8( blockC03 ) );    coordDstTemp.y += 8;

    coordDstTemp = coordDst;    coordDstTemp.x += 8 * sizeof(uint);
    intel_sub_group_block_write8( dst, coordDstTemp, as_uint8( blockC10 ) );    coordDstTemp.y += 8;
    intel_sub_group_block_write8( dst, coordDstTemp, as_uint8( blockC11 ) );    coordDstTemp.y += 8;
    intel_sub_group_block_write8( dst, coordDstTemp, as_uint8( blockC12 ) );    coordDstTemp.y += 8;
    intel_sub_group_block_write8( dst, coordDstTemp, as_uint8( blockC13 ) );    coordDstTemp.y += 8;
}

#undef VEC_SIZE
#undef TILE_M
#undef TILE_K
#undef TILE_N

#endif



////////////////////////////////////////////////////////////////
// MediaBlockRead_SIMD_1x16_2_fp16

#if defined(cl_khr_fp16) && defined(cl_intel_subgroups) && (defined(cl_intel_subgroups_short) || defined(cl_intel_subgroups_half)) && ( defined(ENABLE_KERNEL_ALL) || defined(ENABLE_KERNEL_MEDIABLOCKREAD_SIMD_1X16_2_FP16) )

#define TILE_M          16
#define TILE_K          16
#define TILE_N          16

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void MediaBlockRead_SIMD_1x16_2_fp16(
    __read_only image2d_t src0,
    __read_only image2d_t src1,
    __global half *dst,
    int width0,
    int width1)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);

    // Result ctile is M rows x N columns
    // M = 16, we have 1 rows of work-items, so we need 16/1 = 16 results down = 2 x float8
    // N = 8, we have 8 columns of work-items, so we need 8/8 = 1 result across

    half16 blockC00 = 0.0f;

    // Src0 is directly used as atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int2    coordA = (int2)( 0, group_y * TILE_M );

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    int2    coordB = (int2)( ( group_x * TILE_N ) * sizeof(half), 0 );

    // Walk ACROSS src0 and DOWN src1:
    do
    {
#define TRANSPOSE_BLOCK_16( _blockA00, _blockA01, _col )   \
        (half16)( sub_group_broadcast( _blockA00.s0, _col ),   \
                  sub_group_broadcast( _blockA00.s1, _col ),   \
                  sub_group_broadcast( _blockA00.s2, _col ),   \
                  sub_group_broadcast( _blockA00.s3, _col ),   \
                  sub_group_broadcast( _blockA00.s4, _col ),   \
                  sub_group_broadcast( _blockA00.s5, _col ),   \
                  sub_group_broadcast( _blockA00.s6, _col ),   \
                  sub_group_broadcast( _blockA00.s7, _col ),   \
                  sub_group_broadcast( _blockA01.s0, _col ),   \
                  sub_group_broadcast( _blockA01.s1, _col ),   \
                  sub_group_broadcast( _blockA01.s2, _col ),   \
                  sub_group_broadcast( _blockA01.s3, _col ),   \
                  sub_group_broadcast( _blockA01.s4, _col ),   \
                  sub_group_broadcast( _blockA01.s5, _col ),   \
                  sub_group_broadcast( _blockA01.s6, _col ),   \
                  sub_group_broadcast( _blockA01.s7, _col ) );

#define MULTIPLY_BLOCKS_16x16( _result, _blockA00, _blockA01, _blockB00, _blockB01 )    \
        {   \
            const half16    acol0 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 0 );    \
            const half16    acol1 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 1 );    \
            const half16    acol2 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 2 );    \
            const half16    acol3 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 3 );    \
            const half16    acol4 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 4 );    \
            const half16    acol5 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 5 );    \
            const half16    acol6 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 6 );    \
            const half16    acol7 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 7 );    \
            const half16    acol8 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 8 );    \
            const half16    acol9 = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 9 );    \
            const half16    acola = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 10 );    \
            const half16    acolb = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 11 );    \
            const half16    acolc = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 12 );    \
            const half16    acold = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 13 );    \
            const half16    acole = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 14 );    \
            const half16    acolf = TRANSPOSE_BLOCK_16( _blockA00, _blockA01, 15 );    \
            _result = mad( (half16)(_blockB00.s0), acol0, _result );      \
            _result = mad( (half16)(_blockB00.s1), acol1, _result );      \
            _result = mad( (half16)(_blockB00.s2), acol2, _result );      \
            _result = mad( (half16)(_blockB00.s3), acol3, _result );      \
            _result = mad( (half16)(_blockB00.s4), acol4, _result );      \
            _result = mad( (half16)(_blockB00.s5), acol5, _result );      \
            _result = mad( (half16)(_blockB00.s6), acol6, _result );      \
            _result = mad( (half16)(_blockB00.s7), acol7, _result );      \
            _result = mad( (half16)(_blockB01.s0), acol8, _result );      \
            _result = mad( (half16)(_blockB01.s1), acol9, _result );      \
            _result = mad( (half16)(_blockB01.s2), acola, _result );      \
            _result = mad( (half16)(_blockB01.s3), acolb, _result );      \
            _result = mad( (half16)(_blockB01.s4), acolc, _result );      \
            _result = mad( (half16)(_blockB01.s5), acold, _result );      \
            _result = mad( (half16)(_blockB01.s6), acole, _result );      \
            _result = mad( (half16)(_blockB01.s7), acolf, _result );      \
        }

        // atile is M rows x K columns
        // M = 16, we have 1 row of work-items, so each work-item must load 16/1 = 16 rows
        // K = 8, we have 8 columns of work-items, so each work-item must load 8/8 = 1 column

        // btile is K rows x N columns
        // K = 8, we have 1 row of work-items, so each work-item must load 8/1 = 8 rows
        // N = 8, we have 8 columns of work-items, so each work-item must load 8/8 = 1 column

        // It's annoying that we have to do this interleaving manually -
        // at some point we should investigate why the region prescheduler
        // isn't doing this for us.

        int2   coordBTemp = coordB;
#if defined(cl_intel_subgroups_short)
        half8  blockB00 = as_half8( intel_sub_group_block_read_us8( src1, coordBTemp ) );    coordBTemp.y += 8;
        half8  blockB01 = as_half8( intel_sub_group_block_read_us8( src1, coordBTemp ) );    coordB.y += TILE_K;
#else
        half8  blockB00 = as_half8( intel_sub_group_block_read8_half( src1, coordBTemp ) );    coordBTemp.y += 8;
        half8  blockB01 = as_half8( intel_sub_group_block_read8_half( src1, coordBTemp ) );    coordB.y += TILE_K;
#endif

        int2   coordATemp = coordA;
#if defined(cl_intel_subgroups_short)
        half8  blockA00 = as_half8( intel_sub_group_block_read_us8( src0, coordATemp ) );    coordATemp.y += 8;
        half8  blockA01 = as_half8( intel_sub_group_block_read_us8( src0, coordATemp ) );    coordA.x += TILE_K * sizeof(half);
#else
        half8  blockA00 = as_half8( intel_sub_group_block_read8_half( src0, coordATemp ) );    coordATemp.y += 8;
        half8  blockA01 = as_half8( intel_sub_group_block_read8_half( src0, coordATemp ) );    coordA.x += TILE_K * sizeof(half);
#endif

        MULTIPLY_BLOCKS_16x16( blockC00, blockA00, blockA01, blockB00, blockB01 );

#undef TRANSPOSE_BLOCK_16
#undef MULTIPLY_BLOCKS_16x16
    }
    while( coordB.y < width0 );

    __global half  *dst_write0 = dst + local_x + ( group_x * ( TILE_N ) ) + ( group_y * TILE_M ) * width1;

    dst_write0[ 0 ] = blockC00.s0; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.s1; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.s2; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.s3; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.s4; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.s5; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.s6; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.s7; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.s8; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.s9; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.sa; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.sb; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.sc; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.sd; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.se; dst_write0 += width1;
    dst_write0[ 0 ] = blockC00.sf; dst_write0 += width1;
}
#undef TILE_M
#undef TILE_K
#undef TILE_N

#endif
