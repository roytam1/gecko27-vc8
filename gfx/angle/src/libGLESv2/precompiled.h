//
// Copyright (c) 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// precompiled.h: Precompiled header file for libGLESv2.

#define GL_APICALL
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#define EGLAPI
#include <EGL/egl.h>

#if _MSC_VER <= 1400
#define _interlockedbittestandreset _interlockedbittestandreset_NAME_CHANGED_TO_AVOID_MSVS2005_ERROR
#define _interlockedbittestandset _interlockedbittestandset_NAME_CHANGED_TO_AVOID_MSVS2005_ERROR
#endif

#include <assert.h>
#include <cstddef>
#include <float.h>
#include <intrin.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <algorithm> // for std::min and std::max
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#if !defined(_MSC_VER) || _MSC_VER >= 1500
#include <unordered_map>
#else
#include <boost/unordered_map.hpp>
#endif
#include <vector>

#include <d3d9.h>
#include <D3D11.h>
#include <dxgi.h>
#include <d3dcompiler.h>

#ifdef _MSC_VER
#include <hash_map>
#endif
