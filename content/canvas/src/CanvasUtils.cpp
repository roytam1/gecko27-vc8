/* -*- Mode: C++; tab-width: 20; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#if defined(TT_MEMUTIL) && defined(_MSC_VER)
#include <omp.h>
#endif

#include <tmmintrin.h>
#include "mozilla/SSE.h"
#include "gfxUtils.h"

#include <stdlib.h>
#include <stdarg.h>

#include "prprf.h"

#include "nsIServiceManager.h"

#include "nsIConsoleService.h"
#include "nsIDOMCanvasRenderingContext2D.h"
#include "nsICanvasRenderingContextInternal.h"
#include "nsIHTMLCollection.h"
#include "mozilla/dom/HTMLCanvasElement.h"
#include "nsIPrincipal.h"

#include "nsGfxCIID.h"

#include "nsTArray.h"

#include "CanvasUtils.h"
#include "mozilla/gfx/Matrix.h"

using namespace mozilla::gfx;

namespace mozilla {
namespace CanvasUtils {

void
DoDrawImageSecurityCheck(dom::HTMLCanvasElement *aCanvasElement,
                         nsIPrincipal *aPrincipal,
                         bool forceWriteOnly,
                         bool CORSUsed)
{
    NS_PRECONDITION(aPrincipal, "Must have a principal here");

    // Callers should ensure that mCanvasElement is non-null before calling this
    if (!aCanvasElement) {
        NS_WARNING("DoDrawImageSecurityCheck called without canvas element!");
        return;
    }

    if (aCanvasElement->IsWriteOnly())
        return;

    // If we explicitly set WriteOnly just do it and get out
    if (forceWriteOnly) {
        aCanvasElement->SetWriteOnly();
        return;
    }

    // No need to do a security check if the image used CORS for the load
    if (CORSUsed)
        return;

    // Ignore document.domain in this check.
    bool subsumes;
    nsresult rv =
        aCanvasElement->NodePrincipal()->SubsumesIgnoringDomain(aPrincipal,
                                                                &subsumes);

    if (NS_SUCCEEDED(rv) && subsumes) {
        // This canvas has access to that image anyway
        return;
    }

    aCanvasElement->SetWriteOnly();
}

bool
CoerceDouble(JS::Value v, double* d)
{
    if (JSVAL_IS_DOUBLE(v)) {
        *d = JSVAL_TO_DOUBLE(v);
    } else if (JSVAL_IS_INT(v)) {
        *d = double(JSVAL_TO_INT(v));
    } else if (JSVAL_IS_VOID(v)) {
        *d = 0.0;
    } else {
        return false;
    }
    return true;
}

void
GetImageData_component(uint8_t* _src, uint8_t* _dst,
                       int32_t width, int32_t height,
                       uint32_t srcStride, uint32_t dstStride)
{
    uint8_t *srcFirst = _src;
    uint8_t *dstFirst = _dst;

#if defined(TT_MEMUTIL) && defined(_MSC_VER)
    int omp_thread_counts = omp_get_max_threads();

#pragma omp parallel for schedule(guided) default(none) \
shared(srcFirst, dstFirst, width, height, srcStride, dstStride, gfxUtils::sUnpremultiplyTable) \
if (omp_thread_counts >= 2 && \
    height >= (int32_t)omp_thread_counts && \
    width * height >= 4096)
#endif // defined(TT_MEMUTIL) && defined(_MSC_VER)
    for (int32_t j = 0; j < height; ++j) {
        uint8_t *src = srcFirst + (srcStride * j);
        uint8_t *dst = dstFirst + (dstStride * j);

        for (int32_t i = 0; i < width; ++i) {
            // XXX Is there some useful swizzle MMX we can use here?
#ifdef IS_LITTLE_ENDIAN
            uint8_t b = *src++;
            uint8_t g = *src++;
            uint8_t r = *src++;
            uint8_t a = *src++;
#else
            uint8_t a = *src++;
            uint8_t r = *src++;
            uint8_t g = *src++;
            uint8_t b = *src++;
#endif
            // Convert to non-premultiplied color
            *dst++ = gfxUtils::sUnpremultiplyTable[a * 256 + r];
            *dst++ = gfxUtils::sUnpremultiplyTable[a * 256 + g];
            *dst++ = gfxUtils::sUnpremultiplyTable[a * 256 + b];
            *dst++ = a;
        }
    }
}

void
PutImageData_component(uint8_t* _src, uint8_t* _dst,
                       int32_t width, int32_t height,
                       uint32_t srcStride, uint32_t dstStride)
{
    uint8_t *srcFirst = _src;
    uint8_t *dstFirst = _dst;

    if (mozilla::supports_ssse3()) {
        static const __m128i msk_alpha = _mm_set1_epi32(0xFF000000);
        static const __m128i sfl_alphaLo = _mm_set_epi8(0x80, 7, 0x80, 7, 0x80, 7, 0x80, 7, 0x80, 3, 0x80, 3, 0x80, 3, 0x80, 3);
        static const __m128i sfl_alphaHi = _mm_set_epi8(0x80, 15, 0x80, 15, 0x80, 15, 0x80, 15, 0x80, 11, 0x80, 11, 0x80, 11, 0x80, 11);
        static const __m128i word_add = _mm_set1_epi16(0x00FF);
        static const __m128i word_mul = _mm_set_epi16(0, 257, 257, 257, 0, 257, 257, 257);
        static const __m128i sfl_bgra = _mm_set_epi8(15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2);

#if defined(TT_MEMUTIL) && defined(_MSC_VER)
        int omp_thread_counts = omp_get_max_threads();

#pragma omp parallel for schedule(guided) default(none) \
shared(srcFirst, dstFirst, width, height, srcStride, dstStride, gfxUtils::sPremultiplyTable) \
if (omp_thread_counts >= 2 && \
    height >= (int32_t)omp_thread_counts && \
    width * height >= 12000)
#endif // defined(TT_MEMUTIL) && defined(_MSC_VER)
        for (int j = 0; j < height; j++) {
            uint8_t *src = srcFirst + (srcStride * j);
            uint8_t *dst = dstFirst + (dstStride * j);
            int32_t i = width;

            while (i >= 1 && ((unsigned)dst & 15)) {
                uint8_t r = *src++;
                uint8_t g = *src++;
                uint8_t b = *src++;
                uint8_t a = *src++;

                // Convert to premultiplied color (losslessly if the input came from getImageData)
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + b];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + g];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + r];
                *dst++ = a;
                i -= 1;
            }

            const int srcMissalignedBytes = ((unsigned)src & 15);

            if (srcMissalignedBytes == 0) {
                while (i >= 4) {
                    __m128i xmb = _mm_load_si128((__m128i*)src);
                    __m128i xmwLo = _mm_unpacklo_epi8(xmb, _mm_setzero_si128());
                    __m128i xmwHi = _mm_unpackhi_epi8(xmb, _mm_setzero_si128());

                    __m128i xmwAlpha = _mm_and_si128(xmb, msk_alpha);
                    __m128i xmwAlphaLo = _mm_shuffle_epi8(xmb, sfl_alphaLo);
                    __m128i xmwAlphaHi = _mm_shuffle_epi8(xmb, sfl_alphaHi);

                    xmwLo = _mm_mullo_epi16(xmwLo, xmwAlphaLo);
                    xmwLo = _mm_adds_epu16(xmwLo, word_add);
                    xmwLo = _mm_mulhi_epu16(xmwLo, word_mul);

                    xmwHi = _mm_mullo_epi16(xmwHi, xmwAlphaHi);
                    xmwHi = _mm_adds_epu16(xmwHi, word_add);
                    xmwHi = _mm_mulhi_epu16(xmwHi, word_mul);

                    __m128i xmRes = _mm_packus_epi16(xmwLo, xmwHi);
                    xmRes = _mm_or_si128(xmRes, xmwAlpha);
                    xmRes = _mm_shuffle_epi8(xmRes, sfl_bgra);
                    _mm_store_si128((__m128i*)dst, xmRes);

                    src += 16;
                    dst += 16;
                    i -= 4;
                }
            } else {
                __m128i xmLoadPre = _mm_load_si128((__m128i*)(src - srcMissalignedBytes));

                while (i >= 4) {
                    __m128i xmLoadNext = _mm_load_si128((__m128i*)(src - srcMissalignedBytes + 16));
                    __m128i xmb;

                    switch (srcMissalignedBytes) {
                    case 1:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 1);
                        break;
                    case 2:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 2);
                        break;
                    case 3:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 3);
                        break;
                    case 4:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 4);
                        break;
                    case 5:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 5);
                        break;
                    case 6:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 6);
                        break;
                    case 7:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 7);
                        break;
                    case 8:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 8);
                        break;
                    case 9:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 9);
                        break;
                    case 10:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 10);
                        break;
                    case 11:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 11);
                        break;
                    case 12:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 12);
                        break;
                    case 13:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 13);
                        break;
                    case 14:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 14);
                        break;
                    case 15:
                        xmb = _mm_alignr_epi8(xmLoadNext, xmLoadPre, 15);
                        break;
                    }
                    xmLoadPre = xmLoadNext;

                    __m128i xmwLo = _mm_unpacklo_epi8(xmb, _mm_setzero_si128());
                    __m128i xmwHi = _mm_unpackhi_epi8(xmb, _mm_setzero_si128());

                    __m128i xmwAlpha = _mm_and_si128(xmb, msk_alpha);
                    __m128i xmwAlphaLo = _mm_shuffle_epi8(xmb, sfl_alphaLo);
                    __m128i xmwAlphaHi = _mm_shuffle_epi8(xmb, sfl_alphaHi);

                    xmwLo = _mm_mullo_epi16(xmwLo, xmwAlphaLo);
                    xmwLo = _mm_adds_epu16(xmwLo, word_add);
                    xmwLo = _mm_mulhi_epu16(xmwLo, word_mul);

                    xmwHi = _mm_mullo_epi16(xmwHi, xmwAlphaHi);
                    xmwHi = _mm_adds_epu16(xmwHi, word_add);
                    xmwHi = _mm_mulhi_epu16(xmwHi, word_mul);

                    __m128i xmRes = _mm_packus_epi16(xmwLo, xmwHi);
                    xmRes = _mm_or_si128(xmRes, xmwAlpha);
                    xmRes = _mm_shuffle_epi8(xmRes, sfl_bgra);
                    _mm_store_si128((__m128i*)dst, xmRes);

                    src += 16;
                    dst += 16;
                    i -= 4;
                }
            }

            while (i >= 1) {
                uint8_t r = *src++;
                uint8_t g = *src++;
                uint8_t b = *src++;
                uint8_t a = *src++;

                // Convert to premultiplied color (losslessly if the input came from getImageData)
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + b];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + g];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + r];
                *dst++ = a;
                i -= 1;
            }
        }
    } else {
#if defined(TT_MEMUTIL) && defined(_MSC_VER)
        int omp_thread_counts = omp_get_max_threads();

#pragma omp parallel for schedule(guided) default(none) \
shared(srcFirst, dstFirst, width, height, srcStride, dstStride, gfxUtils::sPremultiplyTable) \
if (omp_thread_counts >= 2 && \
    height >= (int32_t)omp_thread_counts && \
    width * height >= 4096)
#endif // defined(TT_MEMUTIL) && defined(_MSC_VER)
        for (int64_t j = 0; j < height; j++) {
            uint8_t *src = srcFirst + (srcStride * j);
            uint8_t *dst = dstFirst + (dstStride * j);

            for (int32_t i = 0; i < width; i++) {
                // XXX Is there some useful swizzle MMX we can use here?
                uint8_t r = *src++;
                uint8_t g = *src++;
                uint8_t b = *src++;
                uint8_t a = *src++;

                // Convert to premultiplied color (losslessly if the input came from getImageData)
#ifdef IS_LITTLE_ENDIAN
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + b];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + g];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + r];
                *dst++ = a;
#else
                *dst++ = a;
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + r];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + g];
                *dst++ = gfxUtils::sPremultiplyTable[a * 256 + b];
#endif
            }
        }
    }
}

} // namespace CanvasUtils
} // namespace mozilla
