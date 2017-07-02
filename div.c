#include <stdio.h>
#include <string.h>
#include <fenv.h>
#include <immintrin.h>

#define DIVIDEND 15299999
#define DIVISOR DIVIDEND

#define REP8 \
    TEMP(0); TEMP(1); TEMP(2); TEMP(3);\
    TEMP(4); TEMP(5); TEMP(6); TEMP(7);
#define REP16 \
    TEMP(0); TEMP(1); TEMP(2); TEMP(3);\
    TEMP(4); TEMP(5); TEMP(6); TEMP(7);\
    TEMP(8); TEMP(9); TEMP(10); TEMP(11);\
    TEMP(12); TEMP(13); TEMP(14); TEMP(15);

const float dividendf = DIVIDEND;
const float half = 127. / 256;
const float dividendfi = 1.0 / DIVIDEND;

// rdtsc

#if defined(__i386__)

static __inline__ unsigned long long rdtsc ( void )
{
    unsigned long long int x;
    __asm__ volatile ( ".byte 0x0f, 0xae, 0xe8, 0x0f, 0x31, 0x0f, 0xae, 0xe8" : "=A" ( x ) );
    return x;
}

#elif defined(__x86_64__)

static __inline__ unsigned long long rdtsc ( void )
{
    unsigned hi, lo;
    __asm__ __volatile__ (
        //"lfence\n"
        "rdtsc\n"
        //"lfence\n"
        : "=a" ( lo ), "=d" ( hi ) );
    return ( ( unsigned long long ) lo ) | ( ( ( unsigned long long ) hi ) << 32 );
}

#endif

// f is always true

static inline void f ( unsigned int* arr, int n )
{
    for ( int i = 0; i < n ; i++ )
    {
        arr[i] = DIVIDEND / arr[i];
        //arr[i] = arr[i] / DIVISOR;
    }
}

// f2 is usually true

static inline void f2 ( unsigned int* arr, int n )
{
    for ( int i = 0; i < n ; i++ )
    {
        arr[i] = ( float ) DIVIDEND / ( float ) arr[i];
        //arr[i] = arr[i] / DIVISOR;
    }
}

// g is for experiment

static inline __m256 _mm256_rcp_ps_precise ( __m256 x )
{
    // reciprocal for 8 floats
    __m256 y = _mm256_rcp_ps ( x );
    __m256 muls;
    // first iteration
    muls = _mm256_mul_ps ( x, _mm256_mul_ps ( y, y ) );
    y = _mm256_sub_ps ( _mm256_add_ps ( y, y ), muls );
    // second iteration
    //muls = _mm256_mul_ps ( x, _mm256_mul_ps ( y, y ) );
    //y = _mm256_sub_ps ( _mm256_add_ps ( y, y ), muls );
    return y;
}

static inline void g8 ( unsigned int* arr )
{
    // arr[0..7] <= dividend/arr[0..7]
    // use avx2
    // convert 8 int to float
    __m256 v0 = _mm256_cvtepi32_ps ( * ( __m256i* ) arr );
    //__m256 v0a = _mm256_cvtepi32_ps ( * ( __m256i* ) ( arr + 8 ) );
    // prepare dividend
    __m256 v1 = _mm256_broadcast_ss ( &dividendf );
    // inv
    //__m256 v2 = _mm256_rcp_ps_precise ( v0 );
    __m256 v2 = _mm256_rcp_ps ( v0 );
    //__m256 v2a = _mm256_rcp_ps ( v0a );
    // mul
    __m256 val = _mm256_mul_ps ( v2, v1 );
    //__m256 vala = _mm256_mul_ps ( v2a, v1 );
    // standardize
    //__m256 v3 = _mm256_broadcast_ss ( &half );
    //val = _mm256_sub_ps ( val, v3 );
    // prepare dividendi
    __m256 v3 = _mm256_broadcast_ss ( &dividendfi );
    // newtonian iteration
    __m256 kk = _mm256_mul_ps ( v0, v3 );
    //__m256 kka = _mm256_mul_ps ( v0a, v3 );
    __m256 muls;
    //__m256 mulsa;
    muls = _mm256_mul_ps ( kk, _mm256_mul_ps ( val, val ) );
    //mulsa = _mm256_mul_ps ( kka, _mm256_mul_ps ( vala, vala ) );
    val = _mm256_sub_ps ( _mm256_add_ps ( val, val ), muls );
    //vala = _mm256_sub_ps ( _mm256_add_ps ( vala, vala ), mulsa );
    // unpack
    * ( __m256i* ) arr = _mm256_cvtps_epi32 ( val );
    //* ( __m256i* ) ( arr + 8 ) = _mm256_cvtps_epi32 ( vala );
}

static inline void g128_s ( unsigned int* arr )
{
    // arr[0..63] <= dividend/arr[0..63]
    // use avx2
#define REPS REP16
    // prepare dividend
    __m256 v1 = _mm256_broadcast_ss ( &dividendf );
    // convert 8 int to float
#define TEMP(INDEX) \
    __m256 v0##INDEX = _mm256_cvtepi32_ps ( * ( __m256i* ) ( arr + 8 * INDEX ) );
    REPS;
#undef TEMP
    // inv
#define TEMP(INDEX) \
    __m256 val##INDEX = _mm256_div_ps ( v1, v0##INDEX );
    REPS;
#undef TEMP
    // unpack
#define TEMP(INDEX) \
    * ( __m256i* ) ( arr + 8 * INDEX ) = _mm256_cvtps_epi32 ( val##INDEX );
    REPS;
#undef TEMP
#undef REPS
}

static inline void g128 ( unsigned int* arr )
{
    // arr[0..63] <= dividend/arr[0..63]
    // use avx2
#define REPS REP16
    // prepare dividend
    __m256 v1 = _mm256_broadcast_ss ( &dividendf );
    // convert 8 int to float
#define TEMP(INDEX) \
    __m256 v0##INDEX = _mm256_cvtepi32_ps ( * ( __m256i* ) ( arr + 8 * INDEX ) );
    REPS;
#undef TEMP
    // inv
#define TEMP(INDEX) \
    __m256 v2##INDEX = _mm256_rcp_ps ( v0##INDEX );
    REPS;
#undef TEMP
    // mul
#define TEMP(INDEX) \
    __m256 val##INDEX = _mm256_mul_ps ( v2##INDEX, v1 );
    REPS;
#undef TEMP
    // prepare dividendi
    __m256 v3 = _mm256_broadcast_ss ( &dividendfi );
    // newtonian iteration
#define TEMP(INDEX) \
    __m256 kk##INDEX = _mm256_mul_ps ( v0##INDEX, v3 );\
    __m256 muls##INDEX;
    REPS;
#undef TEMP
#define TEMP(INDEX) \
    muls##INDEX = _mm256_mul_ps ( kk##INDEX, _mm256_mul_ps ( val##INDEX, val##INDEX ) );\
    val##INDEX = _mm256_sub_ps ( _mm256_add_ps ( val##INDEX, val##INDEX ), muls##INDEX );
    REPS;
    //REPS;
#undef TEMP
    // unpack
#define TEMP(INDEX) \
    * ( __m256i* ) ( arr + 8 * INDEX ) = _mm256_cvtps_epi32 ( val##INDEX );
    REPS;
#undef TEMP
#undef REPS
}

static inline void g ( unsigned int* arr, int n )
{
    // store the original rounding mode
    const int originalRounding = fegetround( );
    // establish the desired rounding mode
    fesetround ( FE_TOWARDZERO );
    // do whatever you need to do ...
    for ( int i = 0; i < n ; i += 128 )
    {
        g128 ( arr + i );
        //arr[i] = DIVIDEND / arr[i];
        //arr[i] = arr[i] / DIVISOR;
    }
    // ... and restore the original mode afterwards
    fesetround ( originalRounding );
}

//static const int n = 256;
#define n 1024
unsigned int __attribute__ ( ( aligned ( 0x1000 ) ) ) buf[n];
unsigned int __attribute__ ( ( aligned ( 0x1000 ) ) ) buf1[n];
unsigned int __attribute__ ( ( aligned ( 0x1000 ) ) ) buf2[n];
unsigned int __attribute__ ( ( aligned ( 0x1000 ) ) ) buf3[n];

int main()
{
    // read value from stdin, something like "seq 65536" is enough
    for ( int i = 0; i < n; i++ )
    {
        unsigned int input;
        scanf ( "%u", &input );
        if ( input == 0 )
        {
            input = ( unsigned int ) - 1;
        }
        buf[i] = input;
    }
    // correctness
    memcpy ( buf1, buf, sizeof ( buf ) );
    memcpy ( buf2, buf, sizeof ( buf ) );
    memcpy ( buf3, buf, sizeof ( buf ) );
    f ( buf1, n );
    f2 ( buf2, n );
    g ( buf3, n );
    for ( int i = 0; i < n; i++ )
    {
        if ( buf1[i] != buf2[i] )
        {
            fprintf ( stderr, "f != f2: f(%1$d) = %2$d whilst f2(%1$d) = %3$d\n", buf[i], buf1[i], buf2[i] );
        }
        if ( buf1[i] != buf3[i] )
        {
            fprintf ( stderr, "f != g: f(%1$d) = %2$d whilst g(%1$d) = %3$d\n", buf[i], buf1[i], buf3[i] );
        }
    }
    // speed
    while ( 1 )
    {
        // warm up cpu to make it work at max (stable) speed :)
        for ( volatile long i = 0; i < 0x10000000ll; i++ ) {}
        // find out efficiency
        unsigned long begin = 0, end = 0;
        unsigned long t1 = -1, t2 = -1, t3 = -1;
        unsigned long t;
        for ( int i = 0; i < 256; i++ )
        {
            memcpy ( buf1, buf, sizeof ( buf ) );
            memcpy ( buf2, buf, sizeof ( buf ) );
            memcpy ( buf3, buf, sizeof ( buf ) );
            begin = rdtsc();
            f ( buf1, n );
            end = rdtsc();
            t = end - begin;
            if ( t < t1 )
            {
                t1 = t;
            }
            begin = rdtsc();
            f2 ( buf2, n );
            end = rdtsc();
            t = end - begin;
            if ( t < t2 )
            {
                t2 = t;
            }
            begin = rdtsc();
            g ( buf3, n );
            end = rdtsc();
            t = end - begin;
            if ( t < t3 )
            {
                t3 = t;
            }
        }
        fprintf ( stderr, "f:  %lu clocks\n", t1 );
        fprintf ( stderr, "f2: %lu clocks\n", t2 );
        fprintf ( stderr, "g:  %lu clocks\n", t3 );
    }
    return 0;
}
