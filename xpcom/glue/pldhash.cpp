/* -*- Mode: C; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*
 * Double hashing implementation.
 */
#if _MSC_VER >= 1400
#include <intrin.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "prbit.h"
#include "pldhash.h"
#include "mozilla/HashFunctions.h"
#include "nsDebug.h"     /* for PR_ASSERT */
#include "nsAlgorithm.h"
#include "mozilla/Likely.h"
#include "mozilla/MemoryReporting.h"

#ifdef PL_DHASHMETER
# if defined MOZILLA_CLIENT && defined DEBUG_XXXbrendan
#  include "nsTraceMalloc.h"
# endif
# define METER(x)       x
#else
# define METER(x)       /* nothing */
#endif

/*
 * The following DEBUG-only code is used to assert that calls to one of
 * table->ops or to an enumerator do not cause re-entry into a call that
 * can mutate the table.  The recursion level is stored in additional
 * space allocated at the end of the entry store to avoid changing
 * PLDHashTable, which could cause issues when mixing DEBUG and
 * non-DEBUG components.
 */
#ifdef DEBUG

#define RECURSION_LEVEL(table_) (*(uint32_t*)(table_->entryStore + \
                                            PL_DHASH_TABLE_SIZE(table_) * \
                                            table_->entrySize))
/*
 * Most callers that assert about the recursion level don't care about
 * this magical value because they are asserting that mutation is
 * allowed (and therefore the level is 0 or 1, depending on whether they
 * incremented it).
 *
 * Only PL_DHashTableFinish needs to allow this special value.
 */
#define IMMUTABLE_RECURSION_LEVEL ((uint32_t)-1)

#define RECURSION_LEVEL_SAFE_TO_FINISH(table_)                                \
    (RECURSION_LEVEL(table_) == 0 ||                                          \
     RECURSION_LEVEL(table_) == IMMUTABLE_RECURSION_LEVEL)

#define ENTRY_STORE_EXTRA                   sizeof(uint32_t)
#define INCREMENT_RECURSION_LEVEL(table_)                                     \
    PR_BEGIN_MACRO                                                            \
        if (RECURSION_LEVEL(table_) != IMMUTABLE_RECURSION_LEVEL)             \
            ++RECURSION_LEVEL(table_);                                        \
    PR_END_MACRO
#define DECREMENT_RECURSION_LEVEL(table_)                                     \
    PR_BEGIN_MACRO                                                            \
        if (RECURSION_LEVEL(table_) != IMMUTABLE_RECURSION_LEVEL) {           \
            MOZ_ASSERT(RECURSION_LEVEL(table_) > 0);                          \
            --RECURSION_LEVEL(table_);                                        \
        }                                                                     \
    PR_END_MACRO

#else

#define ENTRY_STORE_EXTRA 0
#define INCREMENT_RECURSION_LEVEL(table_)   PR_BEGIN_MACRO PR_END_MACRO
#define DECREMENT_RECURSION_LEVEL(table_)   PR_BEGIN_MACRO PR_END_MACRO

#endif /* defined(DEBUG) */

using namespace mozilla;

void *
PL_DHashAllocTable(PLDHashTable *table, uint32_t nbytes)
{
    return malloc(nbytes);
}

void
PL_DHashFreeTable(PLDHashTable *table, void *ptr)
{
    free(ptr);
}

PLDHashNumber
PL_DHashStringKey(PLDHashTable *table, const void *key)
{
    return HashString(static_cast<const char*>(key));
}

PLDHashNumber
PL_DHashVoidPtrKeyStub(PLDHashTable *table, const void *key)
{
    return (PLDHashNumber)(ptrdiff_t)key >> 2;
}

bool
PL_DHashMatchEntryStub(PLDHashTable *table,
                       const PLDHashEntryHdr *entry,
                       const void *key)
{
    const PLDHashEntryStub *stub = (const PLDHashEntryStub *)entry;

    return stub->key == key;
}

bool
PL_DHashMatchStringKey(PLDHashTable *table,
                       const PLDHashEntryHdr *entry,
                       const void *key)
{
    const PLDHashEntryStub *stub = (const PLDHashEntryStub *)entry;

    /* XXX tolerate null keys on account of sloppy Mozilla callers. */
    return stub->key == key ||
           (stub->key && key &&
            strcmp((const char *) stub->key, (const char *) key) == 0);
}

void
PL_DHashMoveEntryStub(PLDHashTable *table,
                      const PLDHashEntryHdr *from,
                      PLDHashEntryHdr *to)
{
#if _MSC_VER >= 1400
    if ((table->entrySize & 3) == 0) {
        __movsd((unsigned long*)to, (unsigned long*)from, table->entrySize >> 2);
    } else {
        memcpy(to, from, table->entrySize);
    }
#else
    memcpy(to, from, table->entrySize);
#endif
}

void
PL_DHashClearEntryStub(PLDHashTable *table, PLDHashEntryHdr *entry)
{
#if _MSC_VER >= 1400
    if ((table->entrySize & 3) == 0) {
        __stosd((unsigned long*)entry, 0, table->entrySize >> 2);
    } else {
        memset(entry, 0, table->entrySize);
    }
#else
    memset(entry, 0, table->entrySize);
#endif
}

void
PL_DHashFreeStringKey(PLDHashTable *table, PLDHashEntryHdr *entry)
{
    const PLDHashEntryStub *stub = (const PLDHashEntryStub *)entry;

    free((void *) stub->key);
    memset(entry, 0, table->entrySize);
}

void
PL_DHashFinalizeStub(PLDHashTable *table)
{
}

static const PLDHashTableOps stub_ops = {
    PL_DHashAllocTable,
    PL_DHashFreeTable,
    PL_DHashVoidPtrKeyStub,
    PL_DHashMatchEntryStub,
    PL_DHashMoveEntryStub,
    PL_DHashClearEntryStub,
    PL_DHashFinalizeStub,
    nullptr
};

const PLDHashTableOps *
PL_DHashGetStubOps(void)
{
    return &stub_ops;
}

static bool
SizeOfEntryStore(uint32_t capacity, uint32_t entrySize, uint32_t *nbytes)
{
    uint64_t nbytes64 = uint64_t(capacity) * uint64_t(entrySize);
    *nbytes = capacity * entrySize;
    return uint64_t(*nbytes) == nbytes64;   // returns false on overflow
}

PLDHashTable *
PL_NewDHashTable(const PLDHashTableOps *ops, void *data, uint32_t entrySize,
                 uint32_t capacity)
{
    PLDHashTable *table;

    table = (PLDHashTable *) malloc(sizeof *table);
    if (!table)
        return nullptr;
    if (!PL_DHashTableInit(table, ops, data, entrySize, capacity)) {
        free(table);
        return nullptr;
    }
    return table;
}

void
PL_DHashTableDestroy(PLDHashTable *table)
{
    PL_DHashTableFinish(table);
    free(table);
}

bool
PL_DHashTableInit(PLDHashTable *table, const PLDHashTableOps *ops, void *data,
                  uint32_t entrySize, uint32_t capacity)
{
    int log2;
    uint32_t nbytes;

#ifdef DEBUG
    if (entrySize > 16 * sizeof(void *)) {
        printf_stderr(
                "pldhash: for the table at address %p, the given entrySize"
                " of %lu definitely favors chaining over double hashing.\n",
                (void *) table,
                (unsigned long) entrySize);
    }
#endif

    table->ops = ops;
    table->data = data;
    if (capacity < PL_DHASH_MIN_SIZE)
        capacity = PL_DHASH_MIN_SIZE;

    PR_CEILING_LOG2(log2, capacity);

    capacity = 1u << log2;
    if (capacity > PL_DHASH_MAX_SIZE)
        return false;
    table->hashShift = PL_DHASH_BITS - log2;
    table->entrySize = entrySize;
    table->entryCount = table->removedCount = 0;
    table->generation = 0;
    if (!SizeOfEntryStore(capacity, entrySize, &nbytes))
        return false;   // overflowed

    table->entryStore = (char *) ops->allocTable(table,
                                                 nbytes + ENTRY_STORE_EXTRA);
    if (!table->entryStore)
        return false;
    memset(table->entryStore, 0, nbytes);
    METER(memset(&table->stats, 0, sizeof table->stats));

#ifdef DEBUG
    RECURSION_LEVEL(table) = 0;
#endif

    return true;
}

/*
 * Compute max and min load numbers (entry counts).
 */
static inline uint32_t MaxLoad(uint32_t size) {
    return size - (size >> 2);  // == size * 0.75
}
static inline uint32_t MinLoad(uint32_t size) {
    return size >> 2;           // == size * 0.25
}

/*
 * Double hashing needs the second hash code to be relatively prime to table
 * size, so we simply make hash2 odd.
 */
#define HASH1(hash0, shift)         ((hash0) >> (shift))
#define HASH2(hash0,log2,shift)     ((((hash0) << (log2)) >> (shift)) | 1)

/*
 * Reserve keyHash 0 for free entries and 1 for removed-entry sentinels.  Note
 * that a removed-entry sentinel need be stored only if the removed entry had
 * a colliding entry added after it.  Therefore we can use 1 as the collision
 * flag in addition to the removed-entry sentinel value.  Multiplicative hash
 * uses the high order bits of keyHash, so this least-significant reservation
 * should not hurt the hash function's effectiveness much.
 *
 * If you change any of these magic numbers, also update PL_DHASH_ENTRY_IS_LIVE
 * in pldhash.h.  It used to be private to pldhash.c, but then became public to
 * assist iterator writers who inspect table->entryStore directly.
 */
#define COLLISION_FLAG              ((PLDHashNumber) 1)
#define MARK_ENTRY_FREE(entry)      ((entry)->keyHash = 0)
#define MARK_ENTRY_REMOVED(entry)   ((entry)->keyHash = 1)
#define ENTRY_IS_REMOVED(entry)     ((entry)->keyHash == 1)
#define ENTRY_IS_LIVE(entry)        PL_DHASH_ENTRY_IS_LIVE(entry)
#define ENSURE_LIVE_KEYHASH(hash0)  if (hash0 < 2) hash0 -= 2; else (void)0

/* Match an entry's keyHash against an unstored one computed from a key. */
#define MATCH_ENTRY_KEYHASH(entry,hash0) \
    (((entry)->keyHash & ~COLLISION_FLAG) == (hash0))

/* Compute the address of the indexed entry in table. */
#define ADDRESS_ENTRY(table, index) \
    ((PLDHashEntryHdr *)((table)->entryStore + (index) * (table)->entrySize))

void
PL_DHashTableFinish(PLDHashTable *table)
{
    char *entryAddr, *entryLimit;
    uint32_t entrySize;
    PLDHashEntryHdr *entry;

#ifdef DEBUG_XXXbrendan
    static FILE *dumpfp = nullptr;
    if (!dumpfp) dumpfp = fopen("/tmp/pldhash.bigdump", "w");
    if (dumpfp) {
#ifdef MOZILLA_CLIENT
        NS_TraceStack(1, dumpfp);
#endif
        PL_DHashTableDumpMeter(table, nullptr, dumpfp);
        fputc('\n', dumpfp);
    }
#endif

    INCREMENT_RECURSION_LEVEL(table);

    /* Call finalize before clearing entries, so it can enumerate them. */
    table->ops->finalize(table);

    /* Clear any remaining live entries. */
    entryAddr = table->entryStore;
    entrySize = table->entrySize;
    entryLimit = entryAddr + PL_DHASH_TABLE_SIZE(table) * entrySize;
    while (entryAddr < entryLimit) {
        entry = (PLDHashEntryHdr *)entryAddr;
        if (ENTRY_IS_LIVE(entry)) {
            METER(table->stats.removeEnums++);
            table->ops->clearEntry(table, entry);
        }
        entryAddr += entrySize;
    }

    DECREMENT_RECURSION_LEVEL(table);
    MOZ_ASSERT(RECURSION_LEVEL_SAFE_TO_FINISH(table));

    /* Free entry storage last. */
    table->ops->freeTable(table, table->entryStore);
}

static PLDHashEntryHdr * PL_DHASH_FASTCALL
SearchTable(PLDHashTable *table, const void *key, PLDHashNumber keyHash,
            PLDHashOperator op)
{
    PLDHashNumber hash1, hash2;
    int hashShift, sizeLog2;
    PLDHashEntryHdr *entry, *firstRemoved;
    PLDHashMatchEntry matchEntry;
    uint32_t sizeMask;

    METER(table->stats.searches++);
    NS_ASSERTION(!(keyHash & COLLISION_FLAG),
                 "!(keyHash & COLLISION_FLAG)");

    /* Compute the primary hash address. */
    hashShift = table->hashShift;
    hash1 = HASH1(keyHash, hashShift);
    entry = ADDRESS_ENTRY(table, hash1);

    /* Miss: return space for a new entry. */
    if (PL_DHASH_ENTRY_IS_FREE(entry)) {
        METER(table->stats.misses++);
        return entry;
    }

    /* Hit: return entry. */
    matchEntry = table->ops->matchEntry;
    if (MATCH_ENTRY_KEYHASH(entry, keyHash) && matchEntry(table, entry, key)) {
        METER(table->stats.hits++);
        return entry;
    }

    /* Collision: double hash. */
    sizeLog2 = PL_DHASH_BITS - table->hashShift;
    hash2 = HASH2(keyHash, sizeLog2, hashShift);
    sizeMask = (1u << sizeLog2) - 1;

    /* Save the first removed entry pointer so PL_DHASH_ADD can recycle it. */
    firstRemoved = nullptr;

    for (;;) {
        if (MOZ_UNLIKELY(ENTRY_IS_REMOVED(entry))) {
            if (!firstRemoved)
                firstRemoved = entry;
        } else {
            if (op == PL_DHASH_ADD)
                entry->keyHash |= COLLISION_FLAG;
        }

        METER(table->stats.steps++);
        hash1 -= hash2;
        hash1 &= sizeMask;

        entry = ADDRESS_ENTRY(table, hash1);
        if (PL_DHASH_ENTRY_IS_FREE(entry)) {
            METER(table->stats.misses++);
            return (firstRemoved && op == PL_DHASH_ADD) ? firstRemoved : entry;
        }

        if (MATCH_ENTRY_KEYHASH(entry, keyHash) &&
            matchEntry(table, entry, key)) {
            METER(table->stats.hits++);
            return entry;
        }
    }

    /* NOTREACHED */
    return nullptr;
}

/*
 * This is a copy of SearchTable, used by ChangeTable, hardcoded to
 *   1. assume |op == PL_DHASH_ADD|,
 *   2. assume that |key| will never match an existing entry, and
 *   3. assume that no entries have been removed from the current table
 *      structure.
 * Avoiding the need for |key| means we can avoid needing a way to map
 * entries to keys, which means callers can use complex key types more
 * easily.
 */
static PLDHashEntryHdr * PL_DHASH_FASTCALL
FindFreeEntry(PLDHashTable *table, PLDHashNumber keyHash)
{
    PLDHashNumber hash1, hash2;
    int hashShift, sizeLog2;
    PLDHashEntryHdr *entry;
    uint32_t sizeMask;

    METER(table->stats.searches++);
    NS_ASSERTION(!(keyHash & COLLISION_FLAG),
                 "!(keyHash & COLLISION_FLAG)");

    /* Compute the primary hash address. */
    hashShift = table->hashShift;
    hash1 = HASH1(keyHash, hashShift);
    entry = ADDRESS_ENTRY(table, hash1);

    /* Miss: return space for a new entry. */
    if (PL_DHASH_ENTRY_IS_FREE(entry)) {
        METER(table->stats.misses++);
        return entry;
    }

    /* Collision: double hash. */
    sizeLog2 = PL_DHASH_BITS - table->hashShift;
    hash2 = HASH2(keyHash, sizeLog2, hashShift);
    sizeMask = (1u << sizeLog2) - 1;

    for (;;) {
        NS_ASSERTION(!ENTRY_IS_REMOVED(entry),
                     "!ENTRY_IS_REMOVED(entry)");
        entry->keyHash |= COLLISION_FLAG;

        METER(table->stats.steps++);
        hash1 -= hash2;
        hash1 &= sizeMask;

        entry = ADDRESS_ENTRY(table, hash1);
        if (PL_DHASH_ENTRY_IS_FREE(entry)) {
            METER(table->stats.misses++);
            return entry;
        }
    }

    /* NOTREACHED */
    return nullptr;
}

static bool
ChangeTable(PLDHashTable *table, int deltaLog2)
{
    int oldLog2, newLog2;
    uint32_t oldCapacity, newCapacity;
    char *newEntryStore, *oldEntryStore, *oldEntryAddr;
    uint32_t entrySize, i, nbytes;
    PLDHashEntryHdr *oldEntry, *newEntry;
    PLDHashMoveEntry moveEntry;
#ifdef DEBUG
    uint32_t recursionLevel;
#endif

    /* Look, but don't touch, until we succeed in getting new entry store. */
    oldLog2 = PL_DHASH_BITS - table->hashShift;
    newLog2 = oldLog2 + deltaLog2;
    oldCapacity = 1u << oldLog2;
    newCapacity = 1u << newLog2;
    if (newCapacity > PL_DHASH_MAX_SIZE)
        return false;
    entrySize = table->entrySize;
    if (!SizeOfEntryStore(newCapacity, entrySize, &nbytes))
        return false;   // overflowed

    newEntryStore = (char *) table->ops->allocTable(table,
                                                    nbytes + ENTRY_STORE_EXTRA);
    if (!newEntryStore)
        return false;

    /* We can't fail from here on, so update table parameters. */
#ifdef DEBUG
    recursionLevel = RECURSION_LEVEL(table);
#endif
    table->hashShift = PL_DHASH_BITS - newLog2;
    table->removedCount = 0;
    table->generation++;

    /* Assign the new entry store to table. */
#ifdef TT_MEMUTIL
    {
        static PRBool initialized = PR_FALSE;
        static PRUint32 dwNonTemporalDataSizeMin = NON_TEMPORAL_STORES_NOT_SUPPORTED;

        if (!initialized) {
            dwNonTemporalDataSizeMin = GetNonTemporalDataSizeMin_tt();
            initialized = PR_TRUE;
        }

        if (nbytes < dwNonTemporalDataSizeMin ||
            NON_TEMPORAL_STORES_NOT_SUPPORTED == dwNonTemporalDataSizeMin)
        {
            memset(newEntryStore, 0, nbytes);
        }
        else
        {
            memset_nontemporal_tt(newEntryStore, 0, nbytes);
        }
    }
#else
    memset(newEntryStore, 0, nbytes);
#endif
    oldEntryAddr = oldEntryStore = table->entryStore;
    table->entryStore = newEntryStore;
    moveEntry = table->ops->moveEntry;
#ifdef DEBUG
    RECURSION_LEVEL(table) = recursionLevel;
#endif

    /* Copy only live entries, leaving removed ones behind. */
    for (i = 0; i < oldCapacity; i++) {
        oldEntry = (PLDHashEntryHdr *)oldEntryAddr;
        if (ENTRY_IS_LIVE(oldEntry)) {
            oldEntry->keyHash &= ~COLLISION_FLAG;
            newEntry = FindFreeEntry(table, oldEntry->keyHash);
            NS_ASSERTION(PL_DHASH_ENTRY_IS_FREE(newEntry),
                         "PL_DHASH_ENTRY_IS_FREE(newEntry)");
            moveEntry(table, oldEntry, newEntry);
            newEntry->keyHash = oldEntry->keyHash;
        }
        oldEntryAddr += entrySize;
    }

    table->ops->freeTable(table, oldEntryStore);
    return true;
}

PLDHashEntryHdr * PL_DHASH_FASTCALL
PL_DHashTableOperate(PLDHashTable *table, const void *key, PLDHashOperator op)
{
    PLDHashNumber keyHash;
    PLDHashEntryHdr *entry;
    uint32_t size;
    int deltaLog2;

    MOZ_ASSERT(op == PL_DHASH_LOOKUP || RECURSION_LEVEL(table) == 0);
    INCREMENT_RECURSION_LEVEL(table);

    keyHash = table->ops->hashKey(table, key);
    keyHash *= PL_DHASH_GOLDEN_RATIO;

    /* Avoid 0 and 1 hash codes, they indicate free and removed entries. */
    ENSURE_LIVE_KEYHASH(keyHash);
    keyHash &= ~COLLISION_FLAG;

    switch (op) {
      case PL_DHASH_LOOKUP:
        METER(table->stats.lookups++);
        entry = SearchTable(table, key, keyHash, op);
        break;

      case PL_DHASH_ADD:
        /*
         * If alpha is >= .75, grow or compress the table.  If key is already
         * in the table, we may grow once more than necessary, but only if we
         * are on the edge of being overloaded.
         */
        size = PL_DHASH_TABLE_SIZE(table);
        if (table->entryCount + table->removedCount >= MaxLoad(size)) {
            /* Compress if a quarter or more of all entries are removed. */
            if (table->removedCount >= size >> 2) {
                METER(table->stats.compresses++);
                deltaLog2 = 0;
            } else {
                METER(table->stats.grows++);
                deltaLog2 = 1;
            }

            /* Grow or compress table, returning null if ChangeTable fails. */
            if (!ChangeTable(table, deltaLog2)) {
                METER(table->stats.addFailures++);
                entry = nullptr;
                break;
            }
        }

        /*
         * Look for entry after possibly growing, so we don't have to add it,
         * then skip it while growing the table and re-add it after.
         */
        entry = SearchTable(table, key, keyHash, op);
        if (!ENTRY_IS_LIVE(entry)) {
            /* Initialize the entry, indicating that it's no longer free. */
            METER(table->stats.addMisses++);
            if (ENTRY_IS_REMOVED(entry)) {
                METER(table->stats.addOverRemoved++);
                table->removedCount--;
                keyHash |= COLLISION_FLAG;
            }
            if (table->ops->initEntry &&
                !table->ops->initEntry(table, entry, key)) {
                /* We haven't claimed entry yet; fail with null return. */
                memset(entry + 1, 0, table->entrySize - sizeof *entry);
                entry = nullptr;
                break;
            }
            entry->keyHash = keyHash;
            table->entryCount++;
        }
        METER(else table->stats.addHits++);
        break;

      case PL_DHASH_REMOVE:
        entry = SearchTable(table, key, keyHash, op);
        if (ENTRY_IS_LIVE(entry)) {
            /* Clear this entry and mark it as "removed". */
            METER(table->stats.removeHits++);
            PL_DHashTableRawRemove(table, entry);

            /* Shrink if alpha is <= .25 and table isn't too small already. */
            size = PL_DHASH_TABLE_SIZE(table);
            if (size > PL_DHASH_MIN_SIZE &&
                table->entryCount <= MinLoad(size)) {
                METER(table->stats.shrinks++);
                (void) ChangeTable(table, -1);
            }
        }
        METER(else table->stats.removeMisses++);
        entry = nullptr;
        break;

      default:
        NS_NOTREACHED("0");
        entry = nullptr;
    }

    DECREMENT_RECURSION_LEVEL(table);

    return entry;
}

void
PL_DHashTableRawRemove(PLDHashTable *table, PLDHashEntryHdr *entry)
{
    PLDHashNumber keyHash;      /* load first in case clearEntry goofs it */

    MOZ_ASSERT(RECURSION_LEVEL(table) != IMMUTABLE_RECURSION_LEVEL);

    NS_ASSERTION(PL_DHASH_ENTRY_IS_LIVE(entry),
                 "PL_DHASH_ENTRY_IS_LIVE(entry)");
    keyHash = entry->keyHash;
    table->ops->clearEntry(table, entry);
    if (keyHash & COLLISION_FLAG) {
        MARK_ENTRY_REMOVED(entry);
        table->removedCount++;
    } else {
        METER(table->stats.removeFrees++);
        MARK_ENTRY_FREE(entry);
    }
    table->entryCount--;
}

uint32_t
PL_DHashTableEnumerate(PLDHashTable *table, PLDHashEnumerator etor, void *arg)
{
    char *entryAddr, *entryLimit;
    uint32_t i, capacity, entrySize, ceiling;
    bool didRemove;
    PLDHashEntryHdr *entry;
    PLDHashOperator op;

    INCREMENT_RECURSION_LEVEL(table);

    entryAddr = table->entryStore;
    entrySize = table->entrySize;
    capacity = PL_DHASH_TABLE_SIZE(table);
    entryLimit = entryAddr + capacity * entrySize;
    i = 0;
    didRemove = false;
    while (entryAddr < entryLimit) {
        entry = (PLDHashEntryHdr *)entryAddr;
        if (ENTRY_IS_LIVE(entry)) {
            op = etor(table, entry, i++, arg);
            if (op & PL_DHASH_REMOVE) {
                METER(table->stats.removeEnums++);
                PL_DHashTableRawRemove(table, entry);
                didRemove = true;
            }
            if (op & PL_DHASH_STOP)
                break;
        }
        entryAddr += entrySize;
    }

    MOZ_ASSERT(!didRemove || RECURSION_LEVEL(table) == 1);

    /*
     * Shrink or compress if a quarter or more of all entries are removed, or
     * if the table is underloaded according to the minimum alpha, and is not
     * minimal-size already.  Do this only if we removed above, so non-removing
     * enumerations can count on stable table->entryStore until the next
     * non-lookup-Operate or removing-Enumerate.
     */
    if (didRemove &&
        (table->removedCount >= capacity >> 2 ||
         (capacity > PL_DHASH_MIN_SIZE &&
          table->entryCount <= MinLoad(capacity)))) {
        METER(table->stats.enumShrinks++);
        capacity = table->entryCount;
        capacity += capacity >> 1;
        if (capacity < PL_DHASH_MIN_SIZE)
            capacity = PL_DHASH_MIN_SIZE;

        PR_CEILING_LOG2(ceiling, capacity);
        ceiling -= PL_DHASH_BITS - table->hashShift;

        (void) ChangeTable(table, ceiling);
    }

    DECREMENT_RECURSION_LEVEL(table);

    return i;
}

struct SizeOfEntryExcludingThisArg
{
    size_t total;
    PLDHashSizeOfEntryExcludingThisFun sizeOfEntryExcludingThis;
    MallocSizeOf mallocSizeOf;
    void *arg;      // the arg passed by the user
};

static PLDHashOperator
SizeOfEntryExcludingThisEnumerator(PLDHashTable *table, PLDHashEntryHdr *hdr,
                                   uint32_t number, void *arg)
{
    SizeOfEntryExcludingThisArg *e = (SizeOfEntryExcludingThisArg *)arg;
    e->total += e->sizeOfEntryExcludingThis(hdr, e->mallocSizeOf, e->arg);
    return PL_DHASH_NEXT;
}

size_t
PL_DHashTableSizeOfExcludingThis(const PLDHashTable *table,
                                 PLDHashSizeOfEntryExcludingThisFun sizeOfEntryExcludingThis,
                                 MallocSizeOf mallocSizeOf,
                                 void *arg /* = nullptr */)
{
    size_t n = 0;
    n += mallocSizeOf(table->entryStore);
    if (sizeOfEntryExcludingThis) {
        SizeOfEntryExcludingThisArg arg2 = { 0, sizeOfEntryExcludingThis, mallocSizeOf, arg };
        PL_DHashTableEnumerate(const_cast<PLDHashTable *>(table),
                               SizeOfEntryExcludingThisEnumerator, &arg2);
        n += arg2.total;
    }
    return n;
}

size_t
PL_DHashTableSizeOfIncludingThis(const PLDHashTable *table,
                                 PLDHashSizeOfEntryExcludingThisFun sizeOfEntryExcludingThis,
                                 MallocSizeOf mallocSizeOf,
                                 void *arg /* = nullptr */)
{
    return mallocSizeOf(table) +
           PL_DHashTableSizeOfExcludingThis(table, sizeOfEntryExcludingThis,
                                            mallocSizeOf, arg);
}

#ifdef DEBUG
void
PL_DHashMarkTableImmutable(PLDHashTable *table)
{
    RECURSION_LEVEL(table) = IMMUTABLE_RECURSION_LEVEL;
}
#endif

#ifdef PL_DHASHMETER
#include <math.h>

void
PL_DHashTableDumpMeter(PLDHashTable *table, PLDHashEnumerator dump, FILE *fp)
{
    char *entryAddr;
    uint32_t entrySize, entryCount;
    int hashShift, sizeLog2;
    uint32_t i, tableSize, sizeMask, chainLen, maxChainLen, chainCount;
    PLDHashNumber hash1, hash2, saveHash1, maxChainHash1, maxChainHash2;
    double sqsum, mean, variance, sigma;
    PLDHashEntryHdr *entry, *probe;

    entryAddr = table->entryStore;
    entrySize = table->entrySize;
    hashShift = table->hashShift;
    sizeLog2 = PL_DHASH_BITS - hashShift;
    tableSize = PL_DHASH_TABLE_SIZE(table);
    sizeMask = (1u << sizeLog2) - 1;
    chainCount = maxChainLen = 0;
    hash2 = 0;
    sqsum = 0;

    for (i = 0; i < tableSize; i++) {
        entry = (PLDHashEntryHdr *)entryAddr;
        entryAddr += entrySize;
        if (!ENTRY_IS_LIVE(entry))
            continue;
        hash1 = HASH1(entry->keyHash & ~COLLISION_FLAG, hashShift);
        saveHash1 = hash1;
        probe = ADDRESS_ENTRY(table, hash1);
        chainLen = 1;
        if (probe == entry) {
            /* Start of a (possibly unit-length) chain. */
            chainCount++;
        } else {
            hash2 = HASH2(entry->keyHash & ~COLLISION_FLAG, sizeLog2,
                          hashShift);
            do {
                chainLen++;
                hash1 -= hash2;
                hash1 &= sizeMask;
                probe = ADDRESS_ENTRY(table, hash1);
            } while (probe != entry);
        }
        sqsum += chainLen * chainLen;
        if (chainLen > maxChainLen) {
            maxChainLen = chainLen;
            maxChainHash1 = saveHash1;
            maxChainHash2 = hash2;
        }
    }

    entryCount = table->entryCount;
    if (entryCount && chainCount) {
        mean = (double)entryCount / chainCount;
        variance = chainCount * sqsum - entryCount * entryCount;
        if (variance < 0 || chainCount == 1)
            variance = 0;
        else
            variance /= chainCount * (chainCount - 1);
        sigma = sqrt(variance);
    } else {
        mean = sigma = 0;
    }

    fprintf(fp, "Double hashing statistics:\n");
    fprintf(fp, "    table size (in entries): %u\n", tableSize);
    fprintf(fp, "          number of entries: %u\n", table->entryCount);
    fprintf(fp, "  number of removed entries: %u\n", table->removedCount);
    fprintf(fp, "         number of searches: %u\n", table->stats.searches);
    fprintf(fp, "             number of hits: %u\n", table->stats.hits);
    fprintf(fp, "           number of misses: %u\n", table->stats.misses);
    fprintf(fp, "      mean steps per search: %g\n", table->stats.searches ?
                                                     (double)table->stats.steps
                                                     / table->stats.searches :
                                                     0.);
    fprintf(fp, "     mean hash chain length: %g\n", mean);
    fprintf(fp, "         standard deviation: %g\n", sigma);
    fprintf(fp, "  maximum hash chain length: %u\n", maxChainLen);
    fprintf(fp, "          number of lookups: %u\n", table->stats.lookups);
    fprintf(fp, " adds that made a new entry: %u\n", table->stats.addMisses);
    fprintf(fp, "adds that recycled removeds: %u\n", table->stats.addOverRemoved);
    fprintf(fp, "   adds that found an entry: %u\n", table->stats.addHits);
    fprintf(fp, "               add failures: %u\n", table->stats.addFailures);
    fprintf(fp, "             useful removes: %u\n", table->stats.removeHits);
    fprintf(fp, "            useless removes: %u\n", table->stats.removeMisses);
    fprintf(fp, "removes that freed an entry: %u\n", table->stats.removeFrees);
    fprintf(fp, "  removes while enumerating: %u\n", table->stats.removeEnums);
    fprintf(fp, "            number of grows: %u\n", table->stats.grows);
    fprintf(fp, "          number of shrinks: %u\n", table->stats.shrinks);
    fprintf(fp, "       number of compresses: %u\n", table->stats.compresses);
    fprintf(fp, "number of enumerate shrinks: %u\n", table->stats.enumShrinks);

    if (dump && maxChainLen && hash2) {
        fputs("Maximum hash chain:\n", fp);
        hash1 = maxChainHash1;
        hash2 = maxChainHash2;
        entry = ADDRESS_ENTRY(table, hash1);
        i = 0;
        do {
            if (dump(table, entry, i++, fp) != PL_DHASH_NEXT)
                break;
            hash1 -= hash2;
            hash1 &= sizeMask;
            entry = ADDRESS_ENTRY(table, hash1);
        } while (PL_DHASH_ENTRY_IS_BUSY(entry));
    }
}
#endif /* PL_DHASHMETER */
