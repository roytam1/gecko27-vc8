/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 4 -*-
 * vim: set ts=8 sts=4 et sw=4 tw=99:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef ds_LifoAlloc_h
#define ds_LifoAlloc_h

#include "mozilla/DebugOnly.h"
#include "mozilla/MathAlgorithms.h"
#include "mozilla/MemoryChecking.h"
#include "mozilla/MemoryReporting.h"
#include "mozilla/PodOperations.h"
#include "mozilla/TemplateLib.h"
#include "mozilla/TypeTraits.h"

// This data structure supports stacky LIFO allocation (mark/release and
// LifoAllocScope). It does not maintain one contiguous segment; instead, it
// maintains a bunch of linked memory segments. In order to prevent malloc/free
// thrashing, unused segments are deallocated when garbage collection occurs.

#include "jsutil.h"

#if defined(_MSC_VER) && _MSC_VER < 1600
#undef static_assert
#define static_assert(a,b)
#endif

namespace js {

namespace detail {

static const size_t LIFO_ALLOC_ALIGN = 8;

JS_ALWAYS_INLINE
char *
AlignPtr(void *orig)
{
    static_assert(mozilla::tl::FloorLog2<LIFO_ALLOC_ALIGN>::value ==
                  mozilla::tl::CeilingLog2<LIFO_ALLOC_ALIGN>::value,
                  "LIFO_ALLOC_ALIGN must be a power of two");

    char *result = (char *) ((uintptr_t(orig) + (LIFO_ALLOC_ALIGN - 1)) & (~LIFO_ALLOC_ALIGN + 1));
    JS_ASSERT(uintptr_t(result) % LIFO_ALLOC_ALIGN == 0);
    return result;
}

// Header for a chunk of memory wrangled by the LifoAlloc.
class BumpChunk
{
    char        *bump;          // start of the available data
    char        *limit;         // end of the data
    BumpChunk   *next_;         // the next BumpChunk
    size_t      bumpSpaceSize;  // size of the data area

    char *headerBase() { return reinterpret_cast<char *>(this); }
    char *bumpBase() const { return limit - bumpSpaceSize; }

    explicit BumpChunk(size_t bumpSpaceSize)
      : bump(reinterpret_cast<char *>(MOZ_THIS_IN_INITIALIZER_LIST()) + sizeof(BumpChunk)),
        limit(bump + bumpSpaceSize),
        next_(nullptr), bumpSpaceSize(bumpSpaceSize)
    {
        JS_ASSERT(bump == AlignPtr(bump));
    }

    void setBump(void *ptr) {
        JS_ASSERT(bumpBase() <= ptr);
        JS_ASSERT(ptr <= limit);
#if defined(DEBUG) || defined(MOZ_HAVE_MEM_CHECKS)
        char* prevBump = bump;
#endif
        bump = static_cast<char *>(ptr);
#ifdef DEBUG
        JS_ASSERT(contains(prevBump));

        // Clobber the now-free space.
        if (prevBump > bump)
            memset(bump, 0xcd, prevBump - bump);
#endif

        // Poison/Unpoison memory that we just free'd/allocated.
#if defined(MOZ_HAVE_MEM_CHECKS)
        if (prevBump > bump)
            MOZ_MAKE_MEM_NOACCESS(bump, prevBump - bump);
        else if (bump > prevBump)
            MOZ_MAKE_MEM_UNDEFINED(prevBump, bump - prevBump);
#endif
    }

  public:
    BumpChunk *next() const { return next_; }
    void setNext(BumpChunk *succ) { next_ = succ; }

    size_t used() const { return bump - bumpBase(); }

    void *start() const { return bumpBase(); }
    void *end() const { return limit; }

    size_t sizeOfIncludingThis(mozilla::MallocSizeOf mallocSizeOf) {
        return mallocSizeOf(this);
    }

    size_t computedSizeOfIncludingThis() {
        return limit - headerBase();
    }

    void resetBump() {
        setBump(headerBase() + sizeof(BumpChunk));
    }

    void *mark() const { return bump; }

    void release(void *mark) {
        JS_ASSERT(contains(mark));
        JS_ASSERT(mark <= bump);
        setBump(mark);
    }

    bool contains(void *mark) const {
        return bumpBase() <= mark && mark <= limit;
    }

    bool canAlloc(size_t n);

    size_t unused() {
        return limit - AlignPtr(bump);
    }

    // Try to perform an allocation of size |n|, return null if not possible.
    JS_ALWAYS_INLINE
    void *tryAlloc(size_t n) {
        char *aligned = AlignPtr(bump);
        char *newBump = aligned + n;

        if (newBump > limit)
            return nullptr;

        // Check for overflow.
        if (JS_UNLIKELY(newBump < bump))
            return nullptr;

        JS_ASSERT(canAlloc(n)); // Ensure consistency between "can" and "try".
        setBump(newBump);
        return aligned;
    }

    void *allocInfallible(size_t n) {
        void *result = tryAlloc(n);
        JS_ASSERT(result);
        return result;
    }

    static BumpChunk *new_(size_t chunkSize);
    static void delete_(BumpChunk *chunk);
};

} // namespace detail

// LIFO bump allocator: used for phase-oriented and fast LIFO allocations.
//
// Note: |latest| is not necessary "last". We leave BumpChunks latent in the
// chain after they've been released to avoid thrashing before a GC.
class LifoAlloc
{
    typedef detail::BumpChunk BumpChunk;

    BumpChunk   *first;
    BumpChunk   *latest;
    BumpChunk   *last;
    size_t      markCount;
    size_t      defaultChunkSize_;
    size_t      curSize_;
    size_t      peakSize_;

    void operator=(const LifoAlloc &) MOZ_DELETE;
    LifoAlloc(const LifoAlloc &) MOZ_DELETE;

    // Return a BumpChunk that can perform an allocation of at least size |n|
    // and add it to the chain appropriately.
    //
    // Side effect: if retval is non-null, |first| and |latest| are initialized
    // appropriately.
    BumpChunk *getOrCreateChunk(size_t n);

    void reset(size_t defaultChunkSize) {
        JS_ASSERT(mozilla::RoundUpPow2(defaultChunkSize) == defaultChunkSize);
        first = latest = last = nullptr;
        defaultChunkSize_ = defaultChunkSize;
        markCount = 0;
        curSize_ = 0;
    }

    // Append unused chunks to the end of this LifoAlloc.
    void appendUnused(BumpChunk *start, BumpChunk *end) {
        JS_ASSERT(start && end);
        if (last)
            last->setNext(start);
        else
            first = latest = start;
        last = end;
    }

    // Append used chunks to the end of this LifoAlloc. We act as if all the
    // chunks in |this| are used, even if they're not, so memory may be wasted.
    void appendUsed(BumpChunk *start, BumpChunk *latest, BumpChunk *end) {
        JS_ASSERT(start && latest &&  end);
        if (last)
            last->setNext(start);
        else
            first = latest = start;
        last = end;
        this->latest = latest;
    }

    void incrementCurSize(size_t size) {
        curSize_ += size;
        if (curSize_ > peakSize_)
            peakSize_ = curSize_;
    }
    void decrementCurSize(size_t size) {
        JS_ASSERT(curSize_ >= size);
        curSize_ -= size;
    }

  public:
    explicit LifoAlloc(size_t defaultChunkSize)
      : peakSize_(0)
    {
        reset(defaultChunkSize);
    }

    // Steal allocated chunks from |other|.
    void steal(LifoAlloc *other) {
        JS_ASSERT(!other->markCount);

        // Copy everything from |other| to |this| except for |peakSize_|, which
        // requires some care.
        size_t oldPeakSize = peakSize_;
        mozilla::PodAssign(this, other);
        peakSize_ = Max(oldPeakSize, curSize_);

        other->reset(defaultChunkSize_);
    }

    // Append all chunks from |other|. They are removed from |other|.
    void transferFrom(LifoAlloc *other);

    // Append unused chunks from |other|. They are removed from |other|.
    void transferUnusedFrom(LifoAlloc *other);

    ~LifoAlloc() { freeAll(); }

    size_t defaultChunkSize() const { return defaultChunkSize_; }

    // Frees all held memory.
    void freeAll();

    static const unsigned HUGE_ALLOCATION = 50 * 1024 * 1024;
    void freeAllIfHugeAndUnused() {
        if (markCount == 0 && curSize_ > HUGE_ALLOCATION)
            freeAll();
    }

    JS_ALWAYS_INLINE
    void *alloc(size_t n) {
        JS_OOM_POSSIBLY_FAIL();

        void *result;
        if (latest && (result = latest->tryAlloc(n)))
            return result;

        if (!getOrCreateChunk(n))
            return nullptr;

        return latest->allocInfallible(n);
    }

    JS_ALWAYS_INLINE
    void *allocInfallible(size_t n) {
        void *result;
        if (latest && (result = latest->tryAlloc(n)))
            return result;

        mozilla::DebugOnly<BumpChunk *> chunk = getOrCreateChunk(n);
        JS_ASSERT(chunk);

        return latest->allocInfallible(n);
    }

    // Ensures that enough space exists to satisfy N bytes worth of
    // allocation requests, not necessarily contiguous. Note that this does
    // not guarantee a successful single allocation of N bytes.
    JS_ALWAYS_INLINE
    bool ensureUnusedApproximate(size_t n) {
        size_t total = 0;
        for (BumpChunk *chunk = latest; chunk; chunk = chunk->next()) {
            total += chunk->unused();
            if (total >= n)
                return true;
        }
        BumpChunk *latestBefore = latest;
        if (!getOrCreateChunk(n))
            return false;
        if (latestBefore)
            latest = latestBefore;
        return true;
    }

    template <typename T>
    T *newArray(size_t count) {
        JS_STATIC_ASSERT(mozilla::IsPod<T>::value);
        return newArrayUninitialized<T>(count);
    }

    // Create an array with uninitialized elements of type |T|.
    // The caller is responsible for initialization.
    template <typename T>
    T *newArrayUninitialized(size_t count) {
        if (count & mozilla::tl::MulOverflowMask<sizeof(T)>::value)
            return nullptr;
        return static_cast<T *>(alloc(sizeof(T) * count));
    }

    class Mark {
        BumpChunk *chunk;
        void *markInChunk;
        friend class LifoAlloc;
        Mark(BumpChunk *chunk, void *markInChunk) : chunk(chunk), markInChunk(markInChunk) {}
      public:
        Mark() : chunk(nullptr), markInChunk(nullptr) {}
    };

    Mark mark() {
        markCount++;
        return latest ? Mark(latest, latest->mark()) : Mark();
    }

    void release(Mark mark) {
        markCount--;
        if (!mark.chunk) {
            latest = first;
            if (latest)
                latest->resetBump();
        } else {
            latest = mark.chunk;
            latest->release(mark.markInChunk);
        }
    }

    void releaseAll() {
        JS_ASSERT(!markCount);
        latest = first;
        if (latest)
            latest->resetBump();
    }

    // Get the total "used" (occupied bytes) count for the arena chunks.
    size_t used() const {
        size_t accum = 0;
        for (BumpChunk *chunk = first; chunk; chunk = chunk->next()) {
            accum += chunk->used();
            if (chunk == latest)
                break;
        }
        return accum;
    }

    // Return true if the LifoAlloc does not currently contain any allocations.
    bool isEmpty() const {
        return !latest || !latest->used();
    }

    // Return the number of bytes remaining to allocate in the current chunk.
    // e.g. How many bytes we can allocate before needing a new block.
    size_t availableInCurrentChunk() const {
        if (!latest)
            return 0;
        return latest->unused();
    }

    // Get the total size of the arena chunks (including unused space).
    size_t sizeOfExcludingThis(mozilla::MallocSizeOf mallocSizeOf) const {
        size_t n = 0;
        for (BumpChunk *chunk = first; chunk; chunk = chunk->next())
            n += chunk->sizeOfIncludingThis(mallocSizeOf);
        return n;
    }

    // Like sizeOfExcludingThis(), but includes the size of the LifoAlloc itself.
    size_t sizeOfIncludingThis(mozilla::MallocSizeOf mallocSizeOf) const {
        return mallocSizeOf(this) + sizeOfExcludingThis(mallocSizeOf);
    }

    // Get the peak size of the arena chunks (including unused space and
    // bookkeeping space).
    size_t peakSizeOfExcludingThis() const { return peakSize_; }

    // Doesn't perform construction; useful for lazily-initialized POD types.
    template <typename T>
    JS_ALWAYS_INLINE
    T *newPod() {
        return static_cast<T *>(alloc(sizeof(T)));
    }

    JS_DECLARE_NEW_METHODS(new_, alloc, JS_ALWAYS_INLINE)

    // A mutable enumeration of the allocated data.
    class Enum
    {
        friend class LifoAlloc;
        friend class detail::BumpChunk;

        LifoAlloc *alloc_;  // The LifoAlloc being traversed.
        BumpChunk *chunk_;  // The current chunk.
        char *position_;    // The current position (must be within chunk_).

        // If there is not enough room in the remaining block for |size|,
        // advance to the next block and update the position.
        void ensureSpaceAndAlignment(size_t size) {
            JS_ASSERT(!empty());
            char *aligned = detail::AlignPtr(position_);
            if (aligned + size > chunk_->end()) {
                chunk_ = chunk_->next();
                position_ = static_cast<char *>(chunk_->start());
            } else {
                position_ = aligned;
            }
            JS_ASSERT(uintptr_t(position_) + size <= uintptr_t(chunk_->end()));
        }

      public:
        Enum(LifoAlloc &alloc)
          : alloc_(&alloc),
            chunk_(alloc.first),
            position_(static_cast<char *>(alloc.first ? alloc.first->start() : nullptr))
        {}

        // Return true if there are no more bytes to enumerate.
        bool empty() {
            return !chunk_ || (chunk_ == alloc_->latest && position_ >= chunk_->mark());
        }

        // Move the read position forward by the size of one T.
        template <typename T>
        void popFront() {
            popFront(sizeof(T));
        }

        // Move the read position forward by |size| bytes.
        void popFront(size_t size) {
            ensureSpaceAndAlignment(size);
            position_ = position_ + size;
        }

        // Update the bytes at the current position with a new value.
        template <typename T>
        void updateFront(const T &t) {
            ensureSpaceAndAlignment(sizeof(T));
            memmove(position_, &t, sizeof(T));
        }

        // Return a pointer to the item at the current position. This
        // returns a pointer to the inline storage, not a copy.
        template <typename T>
        T *get(size_t size = sizeof(T)) {
            ensureSpaceAndAlignment(size);
            return reinterpret_cast<T *>(position_);
        }

        // Return a Mark at the current position of the Enum.
        Mark mark() {
            alloc_->markCount++;
            return Mark(chunk_, position_);
        }
    };
};

class LifoAllocScope
{
    LifoAlloc       *lifoAlloc;
    LifoAlloc::Mark mark;
    bool            shouldRelease;
    MOZ_DECL_USE_GUARD_OBJECT_NOTIFIER

  public:
    explicit LifoAllocScope(LifoAlloc *lifoAlloc
                            MOZ_GUARD_OBJECT_NOTIFIER_PARAM)
      : lifoAlloc(lifoAlloc),
        mark(lifoAlloc->mark()),
        shouldRelease(true)
    {
        MOZ_GUARD_OBJECT_NOTIFIER_INIT;
    }

    ~LifoAllocScope() {
        if (shouldRelease)
            lifoAlloc->release(mark);
    }

    LifoAlloc &alloc() {
        return *lifoAlloc;
    }

    void releaseEarly() {
        JS_ASSERT(shouldRelease);
        lifoAlloc->release(mark);
        shouldRelease = false;
    }
};

} // namespace js

#endif /* ds_LifoAlloc_h */
