// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/util/defines.h"
#include "seal/util/iterator.h"
#include "seal/util/pointer.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/uintcore.h"
#include <stdexcept>

namespace seal
{
    namespace util
    {
        /**
        Provides an interface to all necessary arithmetic of the number structure that specializes a DWTHandler.
        */
        template <typename ValueType, typename RootType, typename ScalarType>
        class Arithmetic
        {
        public:
            ValueType add(const ValueType &a, const ValueType &b) const;

            ValueType sub(const ValueType &a, const ValueType &b) const;

            ValueType mul_root(const ValueType &a, const RootType &r) const;

            ValueType mul_scalar(const ValueType &a, const ScalarType &s) const;

            RootType mul_root_scalar(const RootType &r, const ScalarType &s) const;

            ValueType guard(const ValueType &a) const;
        };

        /**
        Provides an interface that performs the fast discrete weighted transform (DWT) and its inverse that are used to
        accelerate polynomial multiplications, batch multiple messages into a single plaintext polynomial. This class
        template is specialized with integer modular arithmetic for DWT over integer quotient rings, and is used in
        polynomial multiplications and BatchEncoder. It is also specialized with double-precision complex arithmetic for
        DWT over the complex field, which is used in CKKSEncoder.

        @par The discrete weighted transform (DWT) is a variantion on the discrete Fourier transform (DFT) over
        arbitrary rings involving weighing the input before transforming it by multiplying element-wise by a weight
        vector, then weighing the output by another vector. The DWT can be used to perform negacyclic convolution on
        vectors just like how the DFT can be used to perform cyclic convolution. The DFT of size n requires a primitive
        n-th root of unity, while the DWT for negacyclic convolution requires a primitive 2n-th root of unity, \psi.
        In the forward DWT, the input is multiplied element-wise with an incrementing power of \psi, the forward DFT
        transform uses the 2n-th primitve root of unity \psi^2, and the output is not weighed. In the backward DWT, the
        input is not weighed, the backward DFT transform uses the 2n-th primitve root of unity \psi^{-2}, and the output
        is multiplied element-wise with an incrementing power of \psi^{-1}.

        @par A fast Fourier transform is an algorithm that computes the DFT or its inverse. The Cooley-Tukey FFT reduces
        the complexity of the DFT from O(n^2) to O(n\log{n}). The DFT can be interpretted as evaluating an (n-1)-degree
        polynomial at incrementing powers of a primitive n-th root of unity, which can be accelerated by FFT algorithms.
        The DWT evaluates incrementing odd powers of a primitive 2n-th root of unity, and can also be accelerated by
        FFT-like algorithms implemented in this class.

        @par Algorithms implemented in this class are based on algorithms 1 and 2 in the paper by Patrick Longa and
        Michael Naehrig (https://eprint.iacr.org/2016/504.pdf) with three modifications. First, we generalize in this
        class the algorithms to DWT over arbitrary rings. Second, the powers of \psi^{-1} used by the IDWT are stored
        in a scrambled order (in contrast to bit-reversed order in paper) to create coalesced memory accesses. Third,
        the multiplication with 1/n in the IDWT is merged to the last iteration, saving n/2 multiplications. Last, we
        unroll the loops to create coalesced memory accesses to input and output vectors. In earlier versions of SEAL,
        the mutiplication with 1/n is done by merging a multiplication of 1/2 in all interations, which is slower than
        the current method on CPUs but more efficient on some hardware architectures.

        @par The order in which the powers of \psi^{-1} used by the IDWT are stored is unnatural but efficient:
        the i-th slot stores the (reverse_bits(i - 1, log_n) + 1)-th power of \psi^{-1}.
        */
        template <typename ValueType, typename RootType, typename ScalarType>
        class DWTHandler
        {
        public:
            DWTHandler()
            {}

            DWTHandler(const Arithmetic<ValueType, RootType, ScalarType> &num_struct) : arithmetic_(num_struct)
            {}

            /**
            Performs in place a fast multiplication with the DWT matrix.
            Accesses to powers of root is coalesced.
            Accesses to values is not coalesced without loop unrolling.

            @param[values] inputs in normal order, outputs in bit-reversed order
            @param[log_n] log 2 of the DWT size
            @param[roots] powers of a root in bit-reversed order
            @param[scalar] an optional scalar that is multiplied to all output values
            */
            void transform_to_rev(
                ValueType *values, int log_n, const RootType *roots, const ScalarType *scalar = nullptr) const
            {
                // constant transform size
                size_t n = size_t(1) << log_n;
                std::cout<<"forward NTT, n = "<<n<<std::endl;
                // registers to hold temporary values
                RootType r;
                ValueType u;
                ValueType v;
                // pointers for faster indexing
                ValueType *x = nullptr;
                ValueType *y = nullptr;
                // variables for indexing
                std::size_t gap = n >> 1;
                std::size_t m = 1;
                //for (; m < (n); m <<= 1)
                for (; m < (n >> 1); m <<= 1)
                {
                    std::size_t offset = 0;
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                u = arithmetic_.guard(*x);
                                v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v);
                                *y++ = arithmetic_.sub(u, v);
                            }
                            offset += gap << 1;
                        }

                    gap >>= 1;
                }

                if (scalar != nullptr)
                {
                    RootType scaled_r;
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        scaled_r = arithmetic_.mul_root_scalar(r, *scalar);
                        u = arithmetic_.mul_scalar(arithmetic_.guard(values[0]), *scalar);
                        v = arithmetic_.mul_root(values[1], scaled_r);
                        values[0] = arithmetic_.add(u, v);
                        values[1] = arithmetic_.sub(u, v);
                        values += 2;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        u = arithmetic_.guard(values[0]);
                        v = arithmetic_.mul_root(values[1], r);
                        values[0] = arithmetic_.add(u, v);
                        values[1] = arithmetic_.sub(u, v);
                        values += 2;
                    }
                }
            }


            void transform_to_rev_special(
                ValueType *values, int log_n, const RootType *roots, const ScalarType *scalar = nullptr) const
            {
#if 0
                // constant transform size
                size_t n = size_t(1) << log_n;
                std::cout<<"forward NTT, n = "<<n<<std::endl;
                // registers to hold temporary values
                RootType r;
                ValueType u;
                ValueType v;
                // pointers for faster indexing
                ValueType *x = nullptr;
                ValueType *y = nullptr;
                // variables for indexing
                std::size_t gap = n >> 1;
                std::size_t m = 1;

                //specialized implementation
                ValueType *x0, *x1, *x2, *x3;
                ValueType dx0, dx1, dx2, dx3;
                x0 = values;
                x1 = values + 1;
                x2 = values + 2;
                x3 = values + 3;

                dx0 = *x0; dx1 = *x1; dx2 = *x2; dx3 = *x3;
                ValueType gx0, p2f2, p1f1, p3f1, p1f3, p3f3;
                gx0 = arithmetic_.guard(dx0);
                p2f2 = arithmetic_.mul_root(dx2, roots[1]);
                p1f1 = arithmetic_.mul_root(dx1, roots[2]);
                p3f1 = arithmetic_.mul_root(dx1, roots[3]);
                p1f3 = arithmetic_.mul_root(dx3, roots[2]);
                p3f3 = arithmetic_.mul_root(dx3, roots[3]);
                
                *x0 = arithmetic_.add(arithmetic_.guard(gx0 + p2f2), arithmetic_.guard(p1f1 + p3f3));
                *x1 = arithmetic_.guard(arithmetic_.sub(gx0, p1f1)) + arithmetic_.guard(arithmetic_.sub(p2f2, p3f3));
                *x2 = arithmetic_.guard(arithmetic_.sub(gx0, p2f2)) + arithmetic_.guard(arithmetic_.add(p3f1, p1f3));
                *x3 = arithmetic_.sub(arithmetic_.guard(arithmetic_.sub(gx0, p2f2)), arithmetic_.guard(arithmetic_.add(p3f1, p1f3)));


#else

#if 0
                // constant transform size
                size_t n = size_t(1) << log_n;
                std::cout<<"forward NTT, n = "<<n<<std::endl;
                // registers to hold temporary values
                RootType r1, r2;
                ValueType u;
                ValueType v;
                // pointers for faster indexing
                ValueType *x0 = nullptr;
                ValueType *x1 = nullptr;
                ValueType *x2 = nullptr;
                ValueType *x3 = nullptr;
                // variables for indexing

                std::size_t gap = n >> 2;
                std::size_t log_gap = log_n - 2;//radix 4
                std::size_t m = 1;
                std::size_t total_r = 1;

                // vars for high radix
                ValueType dx0,dx1,dx2,dx3;
                //for (; m < (n >> 1); m <<= 1)
                
                for (; m < (n);total_r += ((m<<1) + m), m <<= 2, gap >>= 2, log_gap -= 2)
                {
                    for (std::size_t ind = 0; ind < (n>>2); ind++){
                        auto ind1 = ((ind >> log_gap) << (log_gap + 1)) + (ind & (gap - 1));
                        auto ind2 = ind1 + gap;
                        auto i1 = (ind1) >> (log_gap+1);
                        auto i2 = (ind2) >> (log_gap+1);
                        auto j = ind1 & ((gap<<1) - 1);
                        auto offset = (i1 << ((log_gap+1) + 1)) + j;
                        r1 = roots[i1+total_r];
                        r2 = roots[i2+total_r];
                        
                        //std::cout<<"r1, gap = "<<gap<<", total_r = "<<total_r<<", {"<<offset<<","<<(offset+gap)<<","<<(offset+gap*2)<<","<<(offset+gap*3)<<"},"<<", ind = "<<ind<<", ind1 = "<<ind1<<", ind2 = "<<ind2<<", r1 = "<<(i1+total_r)<<", r2 = "<<(i2+total_r)<<std::endl;
                        x0 =  values + offset;
                        x1 = x0 + gap;
                        x2 = x1 + gap;
                        x3 = x2 + gap;

                        dx0 = *x0; dx1 = *x1; dx2 = *x2; dx3 = *x3;
                        // inner round 1
                        u = arithmetic_.guard(dx0);
                        v = arithmetic_.mul_root(dx2, r1);
                        dx0 = arithmetic_.add(u, v);
                        dx2 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx1);
                        v = arithmetic_.mul_root(dx3, r2);
                        dx1 = arithmetic_.add(u, v);
                        dx3 = arithmetic_.sub(u, v);

                        // inner round 2
                        i1 = (ind1) >> log_gap;
                        i2 = (ind2) >> log_gap;
                        r1 = roots[i1+total_r+m];
                        r2 = roots[i2+total_r+m];
                        //std::cout<<"round 2, ind = "<<ind<<", total_r = "<<total_r<<", r1 = "<<(i1+total_r+m)<<", r2 = "<<(i2+total_r+m)<<std::endl;
                        u = arithmetic_.guard(dx0);
                        v = arithmetic_.mul_root(dx1, r1);
                        *x0 = arithmetic_.add(u, v);
                        *x1 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx2);
                        v = arithmetic_.mul_root(dx3, r2);
                        *x2 = arithmetic_.add(u, v);
                        *x3 = arithmetic_.sub(u, v);
                    }
                }

#else

                size_t n = size_t(1) << log_n;
                std::cout<<"forward NTT, n = "<<n<<std::endl;
                // registers to hold temporary values
                RootType r0, r1, r2, r3;
                ValueType u;
                ValueType v;
                // pointers for faster indexing
                ValueType *x0 = nullptr;
                ValueType *x1 = nullptr;
                ValueType *x2 = nullptr;
                ValueType *x3 = nullptr;
                ValueType *x4 = nullptr;
                ValueType *x5 = nullptr;
                ValueType *x6 = nullptr;
                ValueType *x7 = nullptr;
                // variables for indexing

                std::size_t gap = n >> 3;
                std::size_t log_gap = log_n - 3;//radix 4
                std::size_t m = 1;
                std::size_t total_r = 1;

                // vars for high radix
                ValueType dx0,dx1,dx2,dx3,dx4,dx5,dx6,dx7;
                //for (; m < (n >> 1); m <<= 1)
                
                for (; m < (n);m <<= 3, total_r = m, gap >>= 3, log_gap -= 3)
                {
                    for (std::size_t ind = 0; ind < (n>>3); ind++){
                        auto ind1 = ((ind >> log_gap) << (log_gap + 2)) + (ind & (gap - 1));
                        auto ind2 = ind1 + gap;
                        auto ind3 = ind2 + gap;
                        auto ind4 = ind3 + gap;
                        auto i1 = (ind1) >> (log_gap+2);
                        auto i2 = (ind2) >> (log_gap+2);
                        auto i3 = (ind3) >> (log_gap+2);
                        auto i4 = (ind4) >> (log_gap+2);
                        auto j = ind1 & ((gap<<2) - 1);
                        auto offset = (i1 << ((log_gap+2) + 1)) + j;
                        r0 = roots[i1+total_r];
                        r1 = roots[i2+total_r];
                        r2 = roots[i3+total_r];
                        r3 = roots[i4+total_r];
                        
                        //std::cout<<"r1, gap = "<<gap<<", total_r = "<<total_r<<", {"<<offset<<","<<(offset+gap)<<","<<(offset+gap*2)<<","<<(offset+gap*3)<<","<<(offset+gap*4)<<","<<(offset+gap*5)<<","<<(offset+gap*6)<<","<<(offset+gap*7)<<"}"<<", ind1{"<<ind1<<","<<ind2<<","<<ind3<<","<<ind4<<"}\n";
                        x0 =  values + offset;
                        x1 = x0 + gap;
                        x2 = x1 + gap;
                        x3 = x2 + gap;

                        x4 = x3 + gap;
                        x5 = x4 + gap;
                        x6 = x5 + gap;
                        x7 = x6 + gap;

                        dx0 = *x0; dx1 = *x1; dx2 = *x2; dx3 = *x3;
                        dx4 = *x4; dx5 = *x5; dx6 = *x6; dx7 = *x7;
                        // inner round 1
                        u = arithmetic_.guard(dx0);
                        v = arithmetic_.mul_root(dx4, r0);
                        dx0 = arithmetic_.add(u, v);
                        dx4 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx1);
                        v = arithmetic_.mul_root(dx5, r1);
                        dx1 = arithmetic_.add(u, v);
                        dx5 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx2);
                        v = arithmetic_.mul_root(dx6, r2);
                        dx2 = arithmetic_.add(u, v);
                        dx6 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx3);
                        v = arithmetic_.mul_root(dx7, r3);
                        dx3 = arithmetic_.add(u, v);
                        dx7 = arithmetic_.sub(u, v);

                        // inner round 2
                        i1 = (ind1) >> (log_gap+1);
                        i2 = (ind2) >> (log_gap+1);
                        i3 = (ind3) >> (log_gap+1);
                        i4 = (ind4) >> (log_gap+1);
                        r0 = roots[i1+(total_r<<1)];
                        r1 = roots[i2+(total_r<<1)];
                        r2 = roots[i3+(total_r<<1)];
                        r3 = roots[i4+(total_r<<1)];
                        u = arithmetic_.guard(dx0);
                        v = arithmetic_.mul_root(dx2, r0);
                        dx0 = arithmetic_.add(u, v);
                        dx2 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx1);
                        v = arithmetic_.mul_root(dx3, r1);
                        dx1 = arithmetic_.add(u, v);
                        dx3 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx4);
                        v = arithmetic_.mul_root(dx6, r2);
                        dx4 = arithmetic_.add(u, v);
                        dx6 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx5);
                        v = arithmetic_.mul_root(dx7, r3);
                        dx5 = arithmetic_.add(u, v);
                        dx7 = arithmetic_.sub(u, v);

                        // inner round 3
                        i1 = (ind1) >> (log_gap);
                        i2 = (ind2) >> (log_gap);
                        i3 = (ind3) >> (log_gap);
                        i4 = (ind4) >> (log_gap);
                        r0 = roots[i1+(total_r<<2)];
                        r1 = roots[i2+(total_r<<2)];
                        r2 = roots[i3+(total_r<<2)];
                        r3 = roots[i4+(total_r<<2)];
                        u = arithmetic_.guard(dx0);
                        v = arithmetic_.mul_root(dx1, r0);
                        *x0 = arithmetic_.add(u, v);
                        *x1 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx2);
                        v = arithmetic_.mul_root(dx3, r1);
                        *x2 = arithmetic_.add(u, v);
                        *x3 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx4);
                        v = arithmetic_.mul_root(dx5, r2);
                        *x4 = arithmetic_.add(u, v);
                        *x5 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx6);
                        v = arithmetic_.mul_root(dx7, r3);
                        *x6 = arithmetic_.add(u, v);
                        *x7 = arithmetic_.sub(u, v);
                    }
                }    

#endif

                // if (scalar != nullptr)
                // {
                //     RootType scaled_r;
                //     for (std::size_t i = 0; i < m; i++)
                //     {
                //         r = *++roots;
                //         scaled_r = arithmetic_.mul_root_scalar(r, *scalar);
                //         u = arithmetic_.mul_scalar(arithmetic_.guard(values[0]), *scalar);
                //         v = arithmetic_.mul_root(values[1], scaled_r);
                //         values[0] = arithmetic_.add(u, v);
                //         values[1] = arithmetic_.sub(u, v);
                //         values += 2;
                //     }
                // }
                // else
                // {
                //     for (std::size_t i = 0; i < m; i++)
                //     {
                //         r = *++roots;
                //         u = arithmetic_.guard(values[0]);
                //         v = arithmetic_.mul_root(values[1], r);
                //         values[0] = arithmetic_.add(u, v);
                //         values[1] = arithmetic_.sub(u, v);
                //         values += 2;
                //     }
                // }
#endif
            }


            /**
            Performs in place a fast multiplication with the DWT matrix.
            Accesses to powers of root is coalesced.
            Accesses to values is not coalesced without loop unrolling.

            @param[values] inputs in bit-reversed order, outputs in normal order
            @param[roots] powers of a root in scrambled order
            @param[scalar] an optional scalar that is multiplied to all output values
            */
            void transform_from_rev(
                ValueType *values, int log_n, const RootType *roots, const ScalarType *scalar = nullptr) const
            {
                
                // constant transform size
                size_t n = size_t(1) << log_n;
                std::cout<<"backward NTT, n = "<<n<<std::endl;
                // registers to hold temporary values
                RootType r;
                ValueType u;
                ValueType v;
                // pointers for faster indexing
                ValueType *x = nullptr;
                ValueType *y = nullptr;
                // variables for indexing
                std::size_t gap = 1;
                std::size_t m = n >> 1;

                for (; m > 1; m >>= 1)
                {
                    std::size_t offset = 0;
                    if (1)
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                            offset += gap << 1;
                        }
                    }
                    else
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j += 4)
                            {
                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                            offset += gap << 1;
                        }
                    }
                    gap <<= 1;
                }

                if (scalar != nullptr)
                {
                    r = *++roots;
                    RootType scaled_r = arithmetic_.mul_root_scalar(r, *scalar);
                    x = values;
                    y = x + gap;
                    if (gap < 4)
                    {
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j < gap; j += 4)
                        {
                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);

                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);

                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);

                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                        }
                    }
                }
                else
                {
                    r = *++roots;
                    x = values;
                    y = x + gap;
                    if (gap < 4)
                    {
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j < gap; j += 4)
                        {
                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                        }
                    }
                }
            }

        private:
            Arithmetic<ValueType, RootType, ScalarType> arithmetic_;
        };
    } // namespace util
} // namespace seal
