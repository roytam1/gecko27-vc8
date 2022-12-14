//
// Copyright (c) 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// InputLayoutCache.h: Defines InputLayoutCache, a class that builds and caches
// D3D11 input layouts.

#ifndef LIBGLESV2_RENDERER_INPUTLAYOUTCACHE_H_
#define LIBGLESV2_RENDERER_INPUTLAYOUTCACHE_H_

#include "libGLESv2/Constants.h"
#include "common/angleutils.h"

namespace gl
{
class ProgramBinary;
}

namespace rx
{
struct TranslatedAttribute;

class InputLayoutCache
{
  public:
    InputLayoutCache();
    virtual ~InputLayoutCache();

    void initialize(ID3D11Device *device, ID3D11DeviceContext *context);
    void clear();
    void markDirty();

    GLenum applyVertexBuffers(TranslatedAttribute attributes[gl::MAX_VERTEX_ATTRIBS],
                              gl::ProgramBinary *programBinary);

  private:
    DISALLOW_COPY_AND_ASSIGN(InputLayoutCache);

    struct InputLayoutKey
    {
        unsigned int elementCount;
        D3D11_INPUT_ELEMENT_DESC elements[gl::MAX_VERTEX_ATTRIBS];
        GLenum glslElementType[gl::MAX_VERTEX_ATTRIBS];
    };

    struct InputLayoutCounterPair
    {
        ID3D11InputLayout *inputLayout;
        unsigned long long lastUsedTime;
    };

    ID3D11InputLayout *mCurrentIL;
    unsigned int mCurrentBuffers[gl::MAX_VERTEX_ATTRIBS];
    UINT mCurrentVertexStrides[gl::MAX_VERTEX_ATTRIBS];
    UINT mCurrentVertexOffsets[gl::MAX_VERTEX_ATTRIBS];

    static std::size_t hashInputLayout(const InputLayoutKey &inputLayout);
    static bool compareInputLayouts(const InputLayoutKey &a, const InputLayoutKey &b);

    typedef std::size_t (*InputLayoutHashFunction)(const InputLayoutKey &);
    typedef bool (*InputLayoutEqualityFunction)(const InputLayoutKey &, const InputLayoutKey &);
#if !defined(_MSC_VER) || _MSC_VER >= 1500
    typedef std::unordered_map<InputLayoutKey,
                               InputLayoutCounterPair,
                               InputLayoutHashFunction,
                               InputLayoutEqualityFunction> InputLayoutMap;
#else
    typedef boost::unordered_map<InputLayoutKey,
                                 InputLayoutCounterPair,
                                 InputLayoutHashFunction,
                                 InputLayoutEqualityFunction> InputLayoutMap;
#endif
    InputLayoutMap mInputLayoutMap;

    static const unsigned int kMaxInputLayouts;

    unsigned long long mCounter;

    ID3D11Device *mDevice;
    ID3D11DeviceContext *mDeviceContext;
};

}

#endif // LIBGLESV2_RENDERER_INPUTLAYOUTCACHE_H_
