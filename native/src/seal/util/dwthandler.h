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

                //for (; m < (n >> 1); m <<= 1)
                for (; m < (n); m <<= 1)
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
            }


            void transform_to_rev_special(
                ValueType *values, int log_n, const RootType *roots, const ScalarType *scalar = nullptr) const
            {
                // constant transform size
                size_t n = size_t(1) << log_n;
                std::cout<<"forward NTT, n = "<<n<<std::endl;
                // registers to hold temporary values
                RootType r[2];
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
                
                for (; m < (n);total_r += m, m <<= 2, gap >>= 2, log_gap -= 2)
                {
                    for (std::size_t ind = 0; ind < (n>>1); ind++){
                        
                        auto i = ind >> (log_gap+1);
                        auto j = ind & ((gap<<1) - 1);
                        auto offset = (i << ((log_gap+1) + 1)) + j;
                        r[0] = roots[i+total_r];
                        r[1] = roots[(ind >> log_gap)+total_r];
                        //std::cout<<"m = "<<m<<", gap = "<<gap<<", total_r = "<<total_r<<", ind = "<<ind<<", i = "<<i<<", j = "<<j<<", x_offset = "<<offset<<", y_offset = "<<(offset+gap)<<", r_op = "<<(total_r+i)<<std::endl;
                        x0 =  values + offset;
                        x1 = x0 + gap;
                        x2 = x1 + gap;
                        x3 = x2 + gap;

                        dx0 = *x0; dx1 = *x1; dx2 = *x2; dx3 = *x3;
                        // inner round 1
                        u = arithmetic_.guard(dx0);
                        v = arithmetic_.mul_root(dx2, r[0]);
                        dx0 = arithmetic_.add(u, v);
                        dx2 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx1);
                        v = arithmetic_.mul_root(dx3, r[0]);
                        dx1 = arithmetic_.add(u, v);
                        dx3 = arithmetic_.sub(u, v);

                        // inner round 2
                        u = arithmetic_.guard(dx0);
                        v = arithmetic_.mul_root(dx1, r[1]);
                        *x0 = arithmetic_.add(u, v);
                        *x1 = arithmetic_.sub(u, v);

                        u = arithmetic_.guard(dx2);
                        v = arithmetic_.mul_root(dx3, r[1]);
                        *x2 = arithmetic_.add(u, v);
                        *x3 = arithmetic_.sub(u, v);
                    }
                }

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
