/* -*- Mode: C++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*-*/
/* vim: set ts=2 sw=2 et tw=79: */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_dom_BindingUtils_h__
#define mozilla_dom_BindingUtils_h__

#include "jsfriendapi.h"
#include "jswrapper.h"
#include "mozilla/Alignment.h"
#include "mozilla/dom/BindingDeclarations.h"
#include "mozilla/dom/CallbackObject.h"
#include "mozilla/dom/DOMJSClass.h"
#include "mozilla/dom/DOMJSProxyHandler.h"
#include "mozilla/dom/Exceptions.h"
#include "mozilla/dom/NonRefcountedDOMObject.h"
#include "mozilla/dom/Nullable.h"
#include "mozilla/dom/RootedDictionary.h"
#include "mozilla/dom/workers/Workers.h"
#include "mozilla/ErrorResult.h"
#include "mozilla/Likely.h"
#include "mozilla/Util.h"
#include "nsCycleCollector.h"
#include "nsIXPConnect.h"
#include "MainThreadUtils.h"
#include "nsTraceRefcnt.h"
#include "qsObjectHelper.h"
#include "xpcpublic.h"
#include "nsIVariant.h"

#include "nsWrapperCacheInlines.h"

#if defined(_MSC_VER) && _MSC_VER < 1600
#undef static_assert
#define static_assert(a,b)
#endif

class nsPIDOMWindow;

extern nsresult
xpc_qsUnwrapArgImpl(JSContext* cx, JS::Handle<JS::Value> v, const nsIID& iid, void** ppArg,
                    nsISupports** ppArgRef, JS::MutableHandle<JS::Value> vp);

namespace mozilla {
namespace dom {

struct SelfRef
{
  SelfRef() : ptr(nullptr) {}
  explicit SelfRef(nsISupports *p) : ptr(p) {}
  ~SelfRef() { NS_IF_RELEASE(ptr); }

  nsISupports* ptr;
};

/** Convert a jsval to an XPCOM pointer. */
template <class Interface, class StrongRefType>
inline nsresult
UnwrapArg(JSContext* cx, JS::Handle<JS::Value> v, Interface** ppArg,
          StrongRefType** ppArgRef, JS::MutableHandle<JS::Value> vp)
{
  nsISupports* argRef = *ppArgRef;
  nsresult rv = xpc_qsUnwrapArgImpl(cx, v, NS_GET_TEMPLATE_IID(Interface),
                                    reinterpret_cast<void**>(ppArg), &argRef,
                                    vp);
  *ppArgRef = static_cast<StrongRefType*>(argRef);
  return rv;
}

bool
ThrowInvalidThis(JSContext* aCx, const JS::CallArgs& aArgs,
                 const ErrNum aErrorNumber,
                 const char* aInterfaceName);

inline bool
ThrowMethodFailedWithDetails(JSContext* cx, ErrorResult& rv,
                             const char* ifaceName,
                             const char* memberName,
                             bool reportJSContentExceptions = false)
{
  if (rv.IsTypeError()) {
    rv.ReportTypeError(cx);
    return false;
  }
  if (rv.IsJSException()) {
    if (reportJSContentExceptions) {
      rv.ReportJSExceptionFromJSImplementation(cx);
    } else {
      rv.ReportJSException(cx);
    }
    return false;
  }
  if (rv.IsNotEnoughArgsError()) {
    rv.ReportNotEnoughArgsError(cx, ifaceName, memberName);
  }
  return Throw(cx, rv.ErrorCode());
}

// Returns true if the JSClass is used for DOM objects.
inline bool
IsDOMClass(const JSClass* clasp)
{
  return clasp->flags & JSCLASS_IS_DOMJSCLASS;
}

inline bool
IsDOMClass(const js::Class* clasp)
{
  return IsDOMClass(Jsvalify(clasp));
}

// Returns true if the JSClass is used for DOM interface and interface 
// prototype objects.
inline bool
IsDOMIfaceAndProtoClass(const JSClass* clasp)
{
  return clasp->flags & JSCLASS_IS_DOMIFACEANDPROTOJSCLASS;
}

inline bool
IsDOMIfaceAndProtoClass(const js::Class* clasp)
{
  return IsDOMIfaceAndProtoClass(Jsvalify(clasp));
}

static_assert(DOM_OBJECT_SLOT == js::PROXY_PRIVATE_SLOT,
              "js::PROXY_PRIVATE_SLOT doesn't match DOM_OBJECT_SLOT.  "
              "Expect bad things");
template <class T>
inline T*
UnwrapDOMObject(JSObject* obj)
{
  MOZ_ASSERT(IsDOMClass(js::GetObjectClass(obj)) || IsDOMProxy(obj),
             "Don't pass non-DOM objects to this function");

  JS::Value val = js::GetReservedSlot(obj, DOM_OBJECT_SLOT);
  return static_cast<T*>(val.toPrivate());
}

inline const DOMClass*
GetDOMClass(JSObject* obj)
{
  const js::Class* clasp = js::GetObjectClass(obj);
  if (IsDOMClass(clasp)) {
    return &DOMJSClass::FromJSClass(clasp)->mClass;
  }

  if (js::IsProxyClass(clasp)) {
    js::BaseProxyHandler* handler = js::GetProxyHandler(obj);
    if (handler->family() == ProxyFamily()) {
      return &static_cast<DOMProxyHandler*>(handler)->mClass;
    }
  }

  return nullptr;
}

inline nsISupports*
UnwrapDOMObjectToISupports(JSObject* aObject)
{
  const DOMClass* clasp = GetDOMClass(aObject);
  if (!clasp || !clasp->mDOMObjectIsISupports) {
    return nullptr;
  }
 
  return UnwrapDOMObject<nsISupports>(aObject);
}

inline bool
IsDOMObject(JSObject* obj)
{
  const js::Class* clasp = js::GetObjectClass(obj);
  return IsDOMClass(clasp) || IsDOMProxy(obj, clasp);
}

#define UNWRAP_OBJECT(Interface, cx, obj, value)                             \
  mozilla::dom::UnwrapObject<mozilla::dom::prototypes::id::Interface,        \
    mozilla::dom::Interface##Binding::NativeType>(cx, obj, value)

// Some callers don't want to set an exception when unwrapping fails
// (for example, overload resolution uses unwrapping to tell what sort
// of thing it's looking at).
// U must be something that a T* can be assigned to (e.g. T* or an nsRefPtr<T>).
template <prototypes::ID PrototypeID, class T, typename U>
MOZ_ALWAYS_INLINE nsresult
UnwrapObject(JSContext* cx, JSObject* obj, U& value)
{
  /* First check to see whether we have a DOM object */
  const DOMClass* domClass = GetDOMClass(obj);
  if (!domClass) {
    /* Maybe we have a security wrapper or outer window? */
    if (!js::IsWrapper(obj)) {
      /* Not a DOM object, not a wrapper, just bail */
      return NS_ERROR_XPC_BAD_CONVERT_JS;
    }

    obj = js::CheckedUnwrap(obj, /* stopAtOuter = */ false);
    if (!obj) {
      return NS_ERROR_XPC_SECURITY_MANAGER_VETO;
    }
    MOZ_ASSERT(!js::IsWrapper(obj));
    domClass = GetDOMClass(obj);
    if (!domClass) {
      /* We don't have a DOM object */
      return NS_ERROR_XPC_BAD_CONVERT_JS;
    }
  }

  /* This object is a DOM object.  Double-check that it is safely
     castable to T by checking whether it claims to inherit from the
     class identified by protoID. */
  if (domClass->mInterfaceChain[PrototypeTraits<PrototypeID>::Depth] ==
      PrototypeID) {
    value = UnwrapDOMObject<T>(obj);
    return NS_OK;
  }

  /* It's the wrong sort of DOM object */
  return NS_ERROR_XPC_BAD_CONVERT_JS;
}

inline bool
IsNotDateOrRegExp(JSContext* cx, JS::Handle<JSObject*> obj)
{
  MOZ_ASSERT(obj);
  return !JS_ObjectIsDate(cx, obj) && !JS_ObjectIsRegExp(cx, obj);
}

MOZ_ALWAYS_INLINE bool
IsArrayLike(JSContext* cx, JS::Handle<JSObject*> obj)
{
  return IsNotDateOrRegExp(cx, obj);
}

MOZ_ALWAYS_INLINE bool
IsObjectValueConvertibleToDictionary(JSContext* cx,
                                     JS::Handle<JS::Value> objVal)
{
  JS::Rooted<JSObject*> obj(cx, &objVal.toObject());
  return IsNotDateOrRegExp(cx, obj);
}

MOZ_ALWAYS_INLINE bool
IsConvertibleToDictionary(JSContext* cx, JS::Handle<JS::Value> val)
{
  return val.isNullOrUndefined() ||
    (val.isObject() && IsObjectValueConvertibleToDictionary(cx, val));
}

MOZ_ALWAYS_INLINE bool
IsConvertibleToCallbackInterface(JSContext* cx, JS::Handle<JSObject*> obj)
{
  return IsNotDateOrRegExp(cx, obj);
}

// The items in the protoAndIfaceArray are indexed by the prototypes::id::ID and
// constructors::id::ID enums, in that order. The end of the prototype objects
// should be the start of the interface objects.
static_assert((size_t)constructors::id::_ID_Start ==
              (size_t)prototypes::id::_ID_Count,
              "Overlapping or discontiguous indexes.");
const size_t kProtoAndIfaceCacheCount = constructors::id::_ID_Count;

inline void
AllocateProtoAndIfaceCache(JSObject* obj)
{
  MOZ_ASSERT(js::GetObjectClass(obj)->flags & JSCLASS_DOM_GLOBAL);
  MOZ_ASSERT(js::GetReservedSlot(obj, DOM_PROTOTYPE_SLOT).isUndefined());

  JS::Heap<JSObject*>* protoAndIfaceArray = new JS::Heap<JSObject*>[kProtoAndIfaceCacheCount];

  js::SetReservedSlot(obj, DOM_PROTOTYPE_SLOT,
                      JS::PrivateValue(protoAndIfaceArray));
}

inline void
TraceProtoAndIfaceCache(JSTracer* trc, JSObject* obj)
{
  MOZ_ASSERT(js::GetObjectClass(obj)->flags & JSCLASS_DOM_GLOBAL);

  if (!HasProtoAndIfaceArray(obj))
    return;
  JS::Heap<JSObject*>* protoAndIfaceArray = GetProtoAndIfaceArray(obj);
  for (size_t i = 0; i < kProtoAndIfaceCacheCount; ++i) {
    if (protoAndIfaceArray[i]) {
      JS_CallHeapObjectTracer(trc, &protoAndIfaceArray[i], "protoAndIfaceArray[i]");
    }
  }
}

inline void
DestroyProtoAndIfaceCache(JSObject* obj)
{
  MOZ_ASSERT(js::GetObjectClass(obj)->flags & JSCLASS_DOM_GLOBAL);

  JS::Heap<JSObject*>* protoAndIfaceArray = GetProtoAndIfaceArray(obj);

  delete [] protoAndIfaceArray;
}

/**
 * Add constants to an object.
 */
bool
DefineConstants(JSContext* cx, JS::Handle<JSObject*> obj,
                const ConstantSpec* cs);

struct JSNativeHolder
{
  JSNative mNative;
  const NativePropertyHooks* mPropertyHooks;
};

struct NamedConstructor
{
  const char* mName;
  const JSNativeHolder mHolder;
  unsigned mNargs;
};

/*
 * Create a DOM interface object (if constructorClass is non-null) and/or a
 * DOM interface prototype object (if protoClass is non-null).
 *
 * global is used as the parent of the interface object and the interface
 *        prototype object
 * protoProto is the prototype to use for the interface prototype object.
 * interfaceProto is the prototype to use for the interface object.
 * protoClass is the JSClass to use for the interface prototype object.
 *            This is null if we should not create an interface prototype
 *            object.
 * protoCache a pointer to a JSObject pointer where we should cache the
 *            interface prototype object. This must be null if protoClass is and
 *            vice versa.
 * constructorClass is the JSClass to use for the interface object.
 *                  This is null if we should not create an interface object or
 *                  if it should be a function object.
 * constructor holds the JSNative to back the interface object which should be a
 *             Function, unless constructorClass is non-null in which case it is
 *             ignored. If this is null and constructorClass is also null then
 *             we should not create an interface object at all.
 * ctorNargs is the length of the constructor function; 0 if no constructor
 * constructorCache a pointer to a JSObject pointer where we should cache the
 *                  interface object. This must be null if both constructorClass
 *                  and constructor are null, and non-null otherwise.
 * domClass is the DOMClass of instance objects for this class.  This can be
 *          null if this is not a concrete proto.
 * properties contains the methods, attributes and constants to be defined on
 *            objects in any compartment.
 * chromeProperties contains the methods, attributes and constants to be defined
 *                  on objects in chrome compartments. This must be null if the
 *                  interface doesn't have any ChromeOnly properties or if the
 *                  object is being created in non-chrome compartment.
 * defineOnGlobal controls whether properties should be defined on the given
 *                global for the interface object (if any) and named
 *                constructors (if any) for this interface.  This can be
 *                false in situations where we want the properties to only
 *                appear on privileged Xrays but not on the unprivileged
 *                underlying global.
 *
 * At least one of protoClass, constructorClass or constructor should be
 * non-null. If constructorClass or constructor are non-null, the resulting
 * interface object will be defined on the given global with property name
 * |name|, which must also be non-null.
 */
void
CreateInterfaceObjects(JSContext* cx, JS::Handle<JSObject*> global,
                       JS::Handle<JSObject*> protoProto,
                       const JSClass* protoClass, JS::Heap<JSObject*>* protoCache,
                       JS::Handle<JSObject*> interfaceProto,
                       const JSClass* constructorClass, const JSNativeHolder* constructor,
                       unsigned ctorNargs, const NamedConstructor* namedConstructors,
                       JS::Heap<JSObject*>* constructorCache, const DOMClass* domClass,
                       const NativeProperties* regularProperties,
                       const NativeProperties* chromeOnlyProperties,
                       const char* name, bool defineOnGlobal);

/*
 * Define the unforgeable attributes on an object.
 */
bool
DefineUnforgeableAttributes(JSContext* cx, JS::Handle<JSObject*> obj,
                            const Prefable<const JSPropertySpec>* props);

bool
DefineWebIDLBindingPropertiesOnXPCProto(JSContext* cx,
                                        JS::Handle<JSObject*> proto,
                                        const NativeProperties* properties);

#ifdef _MSC_VER
#define HAS_MEMBER_CHECK(_name)                                           \
  template<typename V> static yes& Check(char (*)[(&V::_name == 0) + 1])
#else
#define HAS_MEMBER_CHECK(_name)                                           \
  template<typename V> static yes& Check(char (*)[sizeof(&V::_name) + 1])
#endif

#define HAS_MEMBER(_name)                                                 \
template<typename T>                                                      \
class Has##_name##Member {                                                \
  typedef char yes[1];                                                    \
  typedef char no[2];                                                     \
  HAS_MEMBER_CHECK(_name);                                                \
  template<typename V> static no& Check(...);                             \
                                                                          \
public:                                                                   \
  static bool const Value = sizeof(Check<T>(nullptr)) == sizeof(yes);     \
};

HAS_MEMBER(WrapObject)

// HasWrapObject<T>::Value will be true if T has a WrapObject member but it's
// not nsWrapperCache::WrapObject.
template<typename T>
struct HasWrapObject
{
private:
  typedef char yes[1];
  typedef char no[2];
  typedef JSObject* (nsWrapperCache::*WrapObject)(JSContext*,
                                                  JS::Handle<JSObject*>);
  template<typename U, U> struct SFINAE;
  template <typename V> static no& Check(SFINAE<WrapObject, &V::WrapObject>*);
  template <typename V> static yes& Check(...);

public:
  static bool const Value = HasWrapObjectMember<T>::Value &&
                            sizeof(Check<T>(nullptr)) == sizeof(yes);
};

#ifdef DEBUG
template <class T, bool isISupports=IsBaseOf<nsISupports, T>::value>
struct
CheckWrapperCacheCast
{
  static bool Check()
  {
    return reinterpret_cast<uintptr_t>(
      static_cast<nsWrapperCache*>(
        reinterpret_cast<T*>(1))) == 1;
  }
};
template <class T>
struct
CheckWrapperCacheCast<T, true>
{
  static bool Check()
  {
    return true;
  }
};
#endif

MOZ_ALWAYS_INLINE bool
CouldBeDOMBinding(void*)
{
  return true;
}

MOZ_ALWAYS_INLINE bool
CouldBeDOMBinding(nsWrapperCache* aCache)
{
  return aCache->IsDOMBinding();
}

// The DOM_OBJECT_SLOT_SOW slot contains a JS::ObjectValue which points to the
// cached system object wrapper (SOW) or JS::UndefinedValue if this class
// doesn't need SOWs.

inline const JS::Value&
GetSystemOnlyWrapperSlot(JSObject* obj)
{
  MOZ_ASSERT(IsDOMClass(js::GetObjectJSClass(obj)) &&
             !(js::GetObjectJSClass(obj)->flags & JSCLASS_DOM_GLOBAL));
  return js::GetReservedSlot(obj, DOM_OBJECT_SLOT_SOW);
}
inline void
SetSystemOnlyWrapperSlot(JSObject* obj, const JS::Value& v)
{
  MOZ_ASSERT(IsDOMClass(js::GetObjectJSClass(obj)) &&
             !(js::GetObjectJSClass(obj)->flags & JSCLASS_DOM_GLOBAL));
  js::SetReservedSlot(obj, DOM_OBJECT_SLOT_SOW, v);
}

inline bool
GetSameCompartmentWrapperForDOMBinding(JSObject*& obj)
{
  const js::Class* clasp = js::GetObjectClass(obj);
  if (dom::IsDOMClass(clasp)) {
    if (!(clasp->flags & JSCLASS_DOM_GLOBAL)) {
      JS::Value v = GetSystemOnlyWrapperSlot(obj);
      if (v.isObject()) {
        obj = &v.toObject();
      }
    }
    return true;
  }
  return IsDOMProxy(obj, clasp);
}

inline void
SetSystemOnlyWrapper(JSObject* obj, nsWrapperCache* cache, JSObject& wrapper)
{
  SetSystemOnlyWrapperSlot(obj, JS::ObjectValue(wrapper));
  cache->SetHasSystemOnlyWrapper();
}

// Make sure to wrap the given string value into the right compartment, as
// needed.
MOZ_ALWAYS_INLINE
bool
MaybeWrapStringValue(JSContext* cx, JS::MutableHandle<JS::Value> rval)
{
  MOZ_ASSERT(rval.isString());
  JSString* str = rval.toString();
  if (JS::GetGCThingZone(str) != js::GetContextZone(cx)) {
    return JS_WrapValue(cx, rval);
  }
  return true;
}

// Make sure to wrap the given object value into the right compartment as
// needed.  This will work correctly, but possibly slowly, on all objects.
MOZ_ALWAYS_INLINE
bool
MaybeWrapObjectValue(JSContext* cx, JS::MutableHandle<JS::Value> rval)
{
  MOZ_ASSERT(rval.isObject());

  JSObject* obj = &rval.toObject();
  if (js::GetObjectCompartment(obj) != js::GetContextCompartment(cx)) {
    return JS_WrapValue(cx, rval);
  }

  // We're same-compartment, but even then we might need to wrap
  // objects specially.  Check for that.
  if (GetSameCompartmentWrapperForDOMBinding(obj)) {
    // We're a new-binding object, and "obj" now points to the right thing
    rval.set(JS::ObjectValue(*obj));
    return true;
  }

  // It's not a WebIDL object.  But it might be an XPConnect one, in which case
  // we may need to outerize here, so make sure to call JS_WrapValue.
  return JS_WrapValue(cx, rval);
}

// Like MaybeWrapObjectValue, but also allows null
MOZ_ALWAYS_INLINE
bool
MaybeWrapObjectOrNullValue(JSContext* cx, JS::MutableHandle<JS::Value> rval)
{
  MOZ_ASSERT(rval.isObjectOrNull());
  if (rval.isNull()) {
    return true;
  }
  return MaybeWrapObjectValue(cx, rval);
}

// Wrapping for objects that are known to not be DOM or XPConnect objects
MOZ_ALWAYS_INLINE
bool
MaybeWrapNonDOMObjectValue(JSContext* cx, JS::MutableHandle<JS::Value> rval)
{
  MOZ_ASSERT(rval.isObject());
  MOZ_ASSERT(!GetDOMClass(&rval.toObject()));
  MOZ_ASSERT(!(js::GetObjectClass(&rval.toObject())->flags &
               JSCLASS_PRIVATE_IS_NSISUPPORTS));

  JSObject* obj = &rval.toObject();
  if (js::GetObjectCompartment(obj) == js::GetContextCompartment(cx)) {
    return true;
  }
  return JS_WrapValue(cx, rval);
}

// Like MaybeWrapNonDOMObjectValue but allows null
MOZ_ALWAYS_INLINE
bool
MaybeWrapNonDOMObjectOrNullValue(JSContext* cx, JS::MutableHandle<JS::Value> rval)
{
  MOZ_ASSERT(rval.isObjectOrNull());
  if (rval.isNull()) {
    return true;
  }
  return MaybeWrapNonDOMObjectValue(cx, rval);
}

// If rval is a gcthing and is not in the compartment of cx, wrap rval
// into the compartment of cx (typically by replacing it with an Xray or
// cross-compartment wrapper around the original object).
MOZ_ALWAYS_INLINE bool
MaybeWrapValue(JSContext* cx, JS::MutableHandle<JS::Value> rval)
{
  if (rval.isString()) {
    return MaybeWrapStringValue(cx, rval);
  }

  if (!rval.isObject()) {
    return true;
  }

  return MaybeWrapObjectValue(cx, rval);
}

static inline void
WrapNewBindingForSameCompartment(JSContext* cx, JSObject* obj, void* value,
                                 JS::MutableHandle<JS::Value> rval)
{
  rval.set(JS::ObjectValue(*obj));
}

static inline void
WrapNewBindingForSameCompartment(JSContext* cx, JSObject* obj,
                                 nsWrapperCache* value,
                                 JS::MutableHandle<JS::Value> rval)
{
  if (value->HasSystemOnlyWrapper()) {
    rval.set(GetSystemOnlyWrapperSlot(obj));
    MOZ_ASSERT(rval.isObject());
  } else {
    rval.set(JS::ObjectValue(*obj));
  }
}

// Create a JSObject wrapping "value", if there isn't one already, and store it
// in rval.  "value" must be a concrete class that implements a
// GetWrapperPreserveColor() which can return its existing wrapper, if any, and
// a WrapObject() which will try to create a wrapper. Typically, this is done by
// having "value" inherit from nsWrapperCache.
template <class T>
MOZ_ALWAYS_INLINE bool
WrapNewBindingObject(JSContext* cx, JS::Handle<JSObject*> scope, T* value,
                     JS::MutableHandle<JS::Value> rval)
{
  MOZ_ASSERT(value);
  JSObject* obj = value->GetWrapperPreserveColor();
  bool couldBeDOMBinding = CouldBeDOMBinding(value);
  if (obj) {
    JS::ExposeObjectToActiveJS(obj);
  } else {
    // Inline this here while we have non-dom objects in wrapper caches.
    if (!couldBeDOMBinding) {
      return false;
    }

    obj = value->WrapObject(cx, scope);
    if (!obj) {
      // At this point, obj is null, so just return false.
      // Callers seem to be testing JS_IsExceptionPending(cx) to
      // figure out whether WrapObject() threw.
      return false;
    }
  }

#ifdef DEBUG
  const DOMClass* clasp = GetDOMClass(obj);
  // clasp can be null if the cache contained a non-DOM object from a
  // different compartment than scope.
  if (clasp) {
    // Some sanity asserts about our object.  Specifically:
    // 1)  If our class claims we're nsISupports, we better be nsISupports
    //     XXXbz ideally, we could assert that reinterpret_cast to nsISupports
    //     does the right thing, but I don't see a way to do it.  :(
    // 2)  If our class doesn't claim we're nsISupports we better be
    //     reinterpret_castable to nsWrapperCache.
    MOZ_ASSERT(clasp, "What happened here?");
    MOZ_ASSERT_IF(clasp->mDOMObjectIsISupports, (IsBaseOf<nsISupports, T>::value));
    MOZ_ASSERT(CheckWrapperCacheCast<T>::Check());
  }

  // When called via XrayWrapper, we end up here while running in the
  // chrome compartment.  But the obj we have would be created in
  // whatever the content compartment is.  So at this point we need to
  // make sure it's correctly wrapped for the compartment of |scope|.
  // cx should already be in the compartment of |scope| here.
  MOZ_ASSERT(js::IsObjectInContextCompartment(scope, cx));
#endif

  bool sameCompartment =
    js::GetObjectCompartment(obj) == js::GetContextCompartment(cx);
  if (sameCompartment && couldBeDOMBinding) {
    WrapNewBindingForSameCompartment(cx, obj, value, rval);
    return true;
  }

  rval.set(JS::ObjectValue(*obj));
  return JS_WrapValue(cx, rval);
}

// Create a JSObject wrapping "value", for cases when "value" is a
// non-wrapper-cached object using WebIDL bindings.  "value" must implement a
// WrapObject() method taking a JSContext and a scope.
template <class T>
inline bool
WrapNewBindingNonWrapperCachedObject(JSContext* cx,
                                     JS::Handle<JSObject*> scopeArg,
                                     T* value,
                                     JS::MutableHandle<JS::Value> rval)
{
  MOZ_ASSERT(value);
  // We try to wrap in the compartment of the underlying object of "scope"
  JS::Rooted<JSObject*> obj(cx);
  {
    // scope for the JSAutoCompartment so that we restore the compartment
    // before we call JS_WrapValue.
    Maybe<JSAutoCompartment> ac;
    // Maybe<Handle> doesn't so much work, and in any case, adding
    // more Maybe (one for a Rooted and one for a Handle) adds more
    // code (and branches!) than just adding a single rooted.
    JS::Rooted<JSObject*> scope(cx, scopeArg);
    if (js::IsWrapper(scope)) {
      scope = js::CheckedUnwrap(scope, /* stopAtOuter = */ false);
      if (!scope)
        return false;
      ac.construct(cx, scope);
    }

    obj = value->WrapObject(cx, scope);
  }

  if (!obj) {
    return false;
  }

  // We can end up here in all sorts of compartments, per above.  Make
  // sure to JS_WrapValue!
  rval.set(JS::ObjectValue(*obj));
  return JS_WrapValue(cx, rval);
}

// Create a JSObject wrapping "value", for cases when "value" is a
// non-wrapper-cached owned object using WebIDL bindings.  "value" must implement a
// WrapObject() method taking a JSContext, a scope, and a boolean outparam that
// is true if the JSObject took ownership
template <class T>
inline bool
WrapNewBindingNonWrapperCachedOwnedObject(JSContext* cx,
                                          JS::Handle<JSObject*> scopeArg,
                                          nsAutoPtr<T>& value,
                                          JS::MutableHandle<JS::Value> rval)
{
  // We do a runtime check on value, because otherwise we might in
  // fact end up wrapping a null and invoking methods on it later.
  if (!value) {
    NS_RUNTIMEABORT("Don't try to wrap null objects");
  }
  // We try to wrap in the compartment of the underlying object of "scope"
  JS::Rooted<JSObject*> obj(cx);
  {
    // scope for the JSAutoCompartment so that we restore the compartment
    // before we call JS_WrapValue.
    Maybe<JSAutoCompartment> ac;
    // Maybe<Handle> doesn't so much work, and in any case, adding
    // more Maybe (one for a Rooted and one for a Handle) adds more
    // code (and branches!) than just adding a single rooted.
    JS::Rooted<JSObject*> scope(cx, scopeArg);
    if (js::IsWrapper(scope)) {
      scope = js::CheckedUnwrap(scope, /* stopAtOuter = */ false);
      if (!scope)
        return false;
      ac.construct(cx, scope);
    }

    bool tookOwnership = false;
    obj = value->WrapObject(cx, scope, &tookOwnership);
    MOZ_ASSERT_IF(obj, tookOwnership);
    if (tookOwnership) {
      value.forget();
    }
  }

  if (!obj) {
    return false;
  }

  // We can end up here in all sorts of compartments, per above.  Make
  // sure to JS_WrapValue!
  rval.set(JS::ObjectValue(*obj));
  return JS_WrapValue(cx, rval);
}

// Helper for smart pointers (nsAutoPtr/nsRefPtr/nsCOMPtr).
template <template <typename> class SmartPtr, typename T>
inline bool
WrapNewBindingNonWrapperCachedObject(JSContext* cx, JS::Handle<JSObject*> scope,
                                     const SmartPtr<T>& value,
                                     JS::MutableHandle<JS::Value> rval)
{
  return WrapNewBindingNonWrapperCachedObject(cx, scope, value.get(), rval);
}

// Only set allowNativeWrapper to false if you really know you need it, if in
// doubt use true. Setting it to false disables security wrappers.
bool
NativeInterface2JSObjectAndThrowIfFailed(JSContext* aCx,
                                         JS::Handle<JSObject*> aScope,
                                         JS::MutableHandle<JS::Value> aRetval,
                                         xpcObjectHelper& aHelper,
                                         const nsIID* aIID,
                                         bool aAllowNativeWrapper);

/**
 * A method to handle new-binding wrap failure, by possibly falling back to
 * wrapping as a non-new-binding object.
 */
template <class T>
MOZ_ALWAYS_INLINE bool
HandleNewBindingWrappingFailure(JSContext* cx, JS::Handle<JSObject*> scope,
                                T* value, JS::MutableHandle<JS::Value> rval)
{
  if (JS_IsExceptionPending(cx)) {
    return false;
  }

  qsObjectHelper helper(value, GetWrapperCache(value));
  return NativeInterface2JSObjectAndThrowIfFailed(cx, scope, rval,
                                                  helper, nullptr, true);
}

// Helper for calling HandleNewBindingWrappingFailure with smart pointers
// (nsAutoPtr/nsRefPtr/nsCOMPtr) or references.
HAS_MEMBER(get)

template <class T, bool isSmartPtr=HasgetMember<T>::Value>
struct HandleNewBindingWrappingFailureHelper
{
  static inline bool Wrap(JSContext* cx, JS::Handle<JSObject*> scope,
                          const T& value, JS::MutableHandle<JS::Value> rval)
  {
    return HandleNewBindingWrappingFailure(cx, scope, value.get(), rval);
  }
};

template <class T>
struct HandleNewBindingWrappingFailureHelper<T, false>
{
  static inline bool Wrap(JSContext* cx, JS::Handle<JSObject*> scope, T& value,
                          JS::MutableHandle<JS::Value> rval)
  {
    return HandleNewBindingWrappingFailure(cx, scope, &value, rval);
  }
};

template<class T>
inline bool
HandleNewBindingWrappingFailure(JSContext* cx, JS::Handle<JSObject*> scope,
                                T& value, JS::MutableHandle<JS::Value> rval)
{
  return HandleNewBindingWrappingFailureHelper<T>::Wrap(cx, scope, value, rval);
}

template<bool Fatal>
inline bool
EnumValueNotFound(JSContext* cx, const jschar* chars, size_t length,
                  const char* type, const char* sourceDescription)
{
  return false;
}

template<>
inline bool
EnumValueNotFound<false>(JSContext* cx, const jschar* chars, size_t length,
                         const char* type, const char* sourceDescription)
{
  // TODO: Log a warning to the console.
  return true;
}

template<>
inline bool
EnumValueNotFound<true>(JSContext* cx, const jschar* chars, size_t length,
                        const char* type, const char* sourceDescription)
{
  NS_LossyConvertUTF16toASCII deflated(static_cast<const PRUnichar*>(chars),
                                       length);
  return ThrowErrorMessage(cx, MSG_INVALID_ENUM_VALUE, sourceDescription,
                           deflated.get(), type);
}


template<bool InvalidValueFatal>
inline int
FindEnumStringIndex(JSContext* cx, JS::Value v, const EnumEntry* values,
                    const char* type, const char* sourceDescription, bool* ok)
{
  // JS_StringEqualsAscii is slow as molasses, so don't use it here.
  JSString* str = JS_ValueToString(cx, v);
  if (!str) {
    *ok = false;
    return 0;
  }
  JS::Anchor<JSString*> anchor(str);
  size_t length;
  const jschar* chars = JS_GetStringCharsAndLength(cx, str, &length);
  if (!chars) {
    *ok = false;
    return 0;
  }
  int i = 0;
  for (const EnumEntry* value = values; value->value; ++value, ++i) {
    if (length != value->length) {
      continue;
    }

    bool equal = true;
    const char* val = value->value;
    for (size_t j = 0; j != length; ++j) {
      if (unsigned(val[j]) != unsigned(chars[j])) {
        equal = false;
        break;
      }
    }

    if (equal) {
      *ok = true;
      return i;
    }
  }

  *ok = EnumValueNotFound<InvalidValueFatal>(cx, chars, length, type,
                                             sourceDescription);
  return -1;
}

inline nsWrapperCache*
GetWrapperCache(const ParentObject& aParentObject)
{
  return aParentObject.mWrapperCache;
}

template<class T>
inline T*
GetParentPointer(T* aObject)
{
  return aObject;
}

inline nsISupports*
GetParentPointer(const ParentObject& aObject)
{
  return aObject.mObject;
}

template<class T>
inline void
ClearWrapper(T* p, nsWrapperCache* cache)
{
  cache->ClearWrapper();
}

template<class T>
inline void
ClearWrapper(T* p, void*)
{
  nsWrapperCache* cache;
  CallQueryInterface(p, &cache);
  ClearWrapper(p, cache);
}

// Attempt to preserve the wrapper, if any, for a Paris DOM bindings object.
// Return true if we successfully preserved the wrapper, or there is no wrapper
// to preserve. In the latter case we don't need to preserve the wrapper, because
// the object can only be obtained by JS once, or they cannot be meaningfully
// owned from the native side.
//
// This operation will return false only for non-nsISupports cycle-collected
// objects, because we cannot determine if they are wrappercached or not.
bool
TryPreserveWrapper(JSObject* obj);

// Can only be called with the immediate prototype of the instance object. Can
// only be called on the prototype of an object known to be a DOM instance.
bool
InstanceClassHasProtoAtDepth(JS::Handle<JSObject*> protoObject, uint32_t protoID,
                             uint32_t depth);

// Only set allowNativeWrapper to false if you really know you need it, if in
// doubt use true. Setting it to false disables security wrappers.
bool
XPCOMObjectToJsval(JSContext* cx, JS::Handle<JSObject*> scope,
                   xpcObjectHelper& helper, const nsIID* iid,
                   bool allowNativeWrapper, JS::MutableHandle<JS::Value> rval);

// Special-cased wrapping for variants
bool
VariantToJsval(JSContext* aCx, JS::Handle<JSObject*> aScope,
               nsIVariant* aVariant, JS::MutableHandle<JS::Value> aRetval);

// Wrap an object "p" which is not using WebIDL bindings yet.  This _will_
// actually work on WebIDL binding objects that are wrappercached, but will be
// much slower than WrapNewBindingObject.  "cache" must either be null or be the
// nsWrapperCache for "p".
template<class T>
inline bool
WrapObject(JSContext* cx, JS::Handle<JSObject*> scope, T* p,
           nsWrapperCache* cache, const nsIID* iid,
           JS::MutableHandle<JS::Value> rval)
{
  if (xpc_FastGetCachedWrapper(cache, scope, rval))
    return true;
  qsObjectHelper helper(p, cache);
  return XPCOMObjectToJsval(cx, scope, helper, iid, true, rval);
}

// A specialization of the above for nsIVariant, because that needs to
// do something different.
template<>
inline bool
WrapObject<nsIVariant>(JSContext* cx, JS::Handle<JSObject*> scope, nsIVariant* p,
                       nsWrapperCache* cache, const nsIID* iid,
                       JS::MutableHandle<JS::Value> rval)
{
  MOZ_ASSERT(iid);
  MOZ_ASSERT(iid->Equals(NS_GET_IID(nsIVariant)));
  return VariantToJsval(cx, scope, p, rval);
}

// Wrap an object "p" which is not using WebIDL bindings yet.  Just like the
// variant that takes an nsWrapperCache above, but will try to auto-derive the
// nsWrapperCache* from "p".
template<class T>
inline bool
WrapObject(JSContext* cx, JS::Handle<JSObject*> scope, T* p, const nsIID* iid,
           JS::MutableHandle<JS::Value> rval)
{
  return WrapObject(cx, scope, p, GetWrapperCache(p), iid, rval);
}

// Just like the WrapObject above, but without requiring you to pick which
// interface you're wrapping as.  This should only be used for objects that have
// classinfo, for which it doesn't matter what IID is used to wrap.
template<class T>
inline bool
WrapObject(JSContext* cx, JS::Handle<JSObject*> scope, T* p,
           JS::MutableHandle<JS::Value> rval)
{
  return WrapObject(cx, scope, p, NULL, rval);
}

// Helper to make it possible to wrap directly out of an nsCOMPtr
template<class T>
inline bool
WrapObject(JSContext* cx, JS::Handle<JSObject*> scope, const nsCOMPtr<T>& p,
           const nsIID* iid, JS::MutableHandle<JS::Value> rval)
{
  return WrapObject(cx, scope, p.get(), iid, rval);
}

// Helper to make it possible to wrap directly out of an nsCOMPtr
template<class T>
inline bool
WrapObject(JSContext* cx, JS::Handle<JSObject*> scope, const nsCOMPtr<T>& p,
           JS::MutableHandle<JS::Value> rval)
{
  return WrapObject(cx, scope, p, NULL, rval);
}

// Helper to make it possible to wrap directly out of an nsRefPtr
template<class T>
inline bool
WrapObject(JSContext* cx, JS::Handle<JSObject*> scope, const nsRefPtr<T>& p,
           const nsIID* iid, JS::MutableHandle<JS::Value> rval)
{
  return WrapObject(cx, scope, p.get(), iid, rval);
}

// Helper to make it possible to wrap directly out of an nsRefPtr
template<class T>
inline bool
WrapObject(JSContext* cx, JS::Handle<JSObject*> scope, const nsRefPtr<T>& p,
           JS::MutableHandle<JS::Value> rval)
{
  return WrapObject(cx, scope, p, NULL, rval);
}

// Specialization to make it easy to use WrapObject in codegen.
template<>
inline bool
WrapObject<JSObject>(JSContext* cx, JS::Handle<JSObject*> scope, JSObject* p,
                     JS::MutableHandle<JS::Value> rval)
{
  rval.set(JS::ObjectOrNullValue(p));
  return true;
}

inline bool
WrapObject(JSContext* cx, JS::Handle<JSObject*> scope, JSObject& p,
           JS::MutableHandle<JS::Value> rval)
{
  rval.set(JS::ObjectValue(p));
  return true;
}

// Given an object "p" that inherits from nsISupports, wrap it and return the
// result.  Null is returned on wrapping failure.  This is somewhat similar to
// WrapObject() above, but does NOT allow Xrays around the result, since we
// don't want those for our parent object.
template<typename T>
static inline JSObject*
WrapNativeISupportsParent(JSContext* cx, JS::Handle<JSObject*> scope, T* p,
                          nsWrapperCache* cache)
{
  qsObjectHelper helper(ToSupports(p), cache);
  JS::Rooted<JS::Value> v(cx);
  return XPCOMObjectToJsval(cx, scope, helper, nullptr, false, &v) ?
         v.toObjectOrNull() :
         nullptr;
}


// Fallback for when our parent is not a WebIDL binding object.
template<typename T, bool isISupports=IsBaseOf<nsISupports, T>::value>
struct WrapNativeParentFallback
{
  static inline JSObject* Wrap(JSContext* cx, JS::Handle<JSObject*> scope,
                               T* parent, nsWrapperCache* cache)
  {
    return nullptr;
  }
};

// Fallback for when our parent is not a WebIDL binding object but _is_ an
// nsISupports object.
template<typename T >
struct WrapNativeParentFallback<T, true >
{
  static inline JSObject* Wrap(JSContext* cx, JS::Handle<JSObject*> scope,
                               T* parent, nsWrapperCache* cache)
  {
    return WrapNativeISupportsParent(cx, scope, parent, cache);
  }
};

// Wrapping of our native parent, for cases when it's a WebIDL object (though
// possibly preffed off).
template<typename T, bool hasWrapObject=HasWrapObject<T>::Value >
struct WrapNativeParentHelper
{
  static inline JSObject* Wrap(JSContext* cx, JS::Handle<JSObject*> scope,
                               T* parent, nsWrapperCache* cache)
  {
    MOZ_ASSERT(cache);

    JSObject* obj;
    if ((obj = cache->GetWrapper())) {
      return obj;
    }

    // Inline this here while we have non-dom objects in wrapper caches.
    if (!CouldBeDOMBinding(parent)) {
      obj = WrapNativeParentFallback<T>::Wrap(cx, scope, parent, cache);
    } else {
      obj = parent->WrapObject(cx, scope);
    }

    return obj;
  }
};

// Wrapping of our native parent, for cases when it's not a WebIDL object.  In
// this case it must be nsISupports.
template<typename T>
struct WrapNativeParentHelper<T, false >
{
  static inline JSObject* Wrap(JSContext* cx, JS::Handle<JSObject*> scope,
                               T* parent, nsWrapperCache* cache)
  {
    JSObject* obj;
    if (cache && (obj = cache->GetWrapper())) {
#ifdef DEBUG
      NS_ASSERTION(WrapNativeISupportsParent(cx, scope, parent, cache) == obj,
                   "Unexpected object in nsWrapperCache");
#endif
      return obj;
    }

    return WrapNativeISupportsParent(cx, scope, parent, cache);
  }
};

// Wrapping of our native parent.
template<typename T>
static inline JSObject*
WrapNativeParent(JSContext* cx, JS::Handle<JSObject*> scope, T* p,
                 nsWrapperCache* cache)
{
  if (!p) {
    return scope;
  }

  return WrapNativeParentHelper<T>::Wrap(cx, scope, p, cache);
}

// Wrapping of our native parent, when we don't want to explicitly pass in
// things like the nsWrapperCache for it.
template<typename T>
static inline JSObject*
WrapNativeParent(JSContext* cx, JS::Handle<JSObject*> scope, const T& p)
{
  return WrapNativeParent(cx, scope, GetParentPointer(p), GetWrapperCache(p));
}

// A way to differentiate between nodes, which use the parent object
// returned by native->GetParentObject(), and all other objects, which
// just use the parent's global.
static inline JSObject*
GetRealParentObject(void* aParent, JSObject* aParentObject)
{
  return aParentObject ?
    js::GetGlobalForObjectCrossCompartment(aParentObject) : nullptr;
}

static inline JSObject*
GetRealParentObject(Element* aParent, JSObject* aParentObject)
{
  return aParentObject;
}

HAS_MEMBER(GetParentObject)

template<typename T, bool WrapperCached=HasGetParentObjectMember<T>::Value>
struct GetParentObject
{
  static JSObject* Get(JSContext* cx, JS::Handle<JSObject*> obj)
  {
    T* native = UnwrapDOMObject<T>(obj);
    return
      GetRealParentObject(native,
                          WrapNativeParent(cx, obj, native->GetParentObject()));
  }
};

template<typename T>
struct GetParentObject<T, false>
{
  static JSObject* Get(JSContext* cx, JS::Handle<JSObject*> obj)
  {
    MOZ_CRASH();
    return nullptr;
  }
};

MOZ_ALWAYS_INLINE
JSObject* GetJSObjectFromCallback(CallbackObject* callback)
{
  return callback->Callback();
}

MOZ_ALWAYS_INLINE
JSObject* GetJSObjectFromCallback(void* noncallback)
{
  return nullptr;
}

template<typename T>
static inline JSObject*
WrapCallThisObject(JSContext* cx, JS::Handle<JSObject*> scope, const T& p)
{
  // Callbacks are nsISupports, so WrapNativeParent will just happily wrap them
  // up as an nsISupports XPCWrappedNative... which is not at all what we want.
  // So we need to special-case them.
  JS::Rooted<JSObject*> obj(cx, GetJSObjectFromCallback(p));
  if (!obj) {
    // WrapNativeParent is a bit of a Swiss army knife that will
    // wrap anything for us.
    obj = WrapNativeParent(cx, scope, p);
    if (!obj) {
      return nullptr;
    }
  }

  // But all that won't necessarily put things in the compartment of cx.
  if (!JS_WrapObject(cx, &obj)) {
    return nullptr;
  }

  return obj;
}

// Helper for calling WrapNewBindingObject with smart pointers
// (nsAutoPtr/nsRefPtr/nsCOMPtr) or references.
template <class T, bool isSmartPtr=HasgetMember<T>::Value>
struct WrapNewBindingObjectHelper
{
  static inline bool Wrap(JSContext* cx, JS::Handle<JSObject*> scope,
                          const T& value, JS::MutableHandle<JS::Value> rval)
  {
    return WrapNewBindingObject(cx, scope, value.get(), rval);
  }
};

template <class T>
struct WrapNewBindingObjectHelper<T, false>
{
  static inline bool Wrap(JSContext* cx, JS::Handle<JSObject*> scope, T& value,
                          JS::MutableHandle<JS::Value> rval)
  {
    return WrapNewBindingObject(cx, scope, &value, rval);
  }
};

template<class T>
inline bool
WrapNewBindingObject(JSContext* cx, JS::Handle<JSObject*> scope, T& value,
                     JS::MutableHandle<JS::Value> rval)
{
  return WrapNewBindingObjectHelper<T>::Wrap(cx, scope, value, rval);
}

template <class T>
inline JSObject*
GetCallbackFromCallbackObject(T* aObj)
{
  return aObj->Callback();
}

// Helper for getting the callback JSObject* of a smart ptr around a
// CallbackObject or a reference to a CallbackObject or something like
// that.
template <class T, bool isSmartPtr=HasgetMember<T>::Value>
struct GetCallbackFromCallbackObjectHelper
{
  static inline JSObject* Get(const T& aObj)
  {
    return GetCallbackFromCallbackObject(aObj.get());
  }
};

template <class T>
struct GetCallbackFromCallbackObjectHelper<T, false>
{
  static inline JSObject* Get(T& aObj)
  {
    return GetCallbackFromCallbackObject(&aObj);
  }
};

template<class T>
inline JSObject*
GetCallbackFromCallbackObject(T& aObj)
{
  return GetCallbackFromCallbackObjectHelper<T>::Get(aObj);
}

static inline bool
InternJSString(JSContext* cx, jsid& id, const char* chars)
{
  if (JSString *str = ::JS_InternString(cx, chars)) {
    id = INTERNED_STRING_TO_JSID(cx, str);
    return true;
  }
  return false;
}

// Spec needs a name property
template <typename Spec>
static bool
InitIds(JSContext* cx, const Prefable<Spec>* prefableSpecs, jsid* ids)
{
  MOZ_ASSERT(prefableSpecs);
  MOZ_ASSERT(prefableSpecs->specs);
  do {
    // We ignore whether the set of ids is enabled and just intern all the IDs,
    // because this is only done once per application runtime.
    Spec* spec = prefableSpecs->specs;
    do {
      if (!InternJSString(cx, *ids, spec->name)) {
        return false;
      }
    } while (++ids, (++spec)->name);

    // We ran out of ids for that pref.  Put a JSID_VOID in on the id
    // corresponding to the list terminator for the pref.
    *ids = JSID_VOID;
    ++ids;
  } while ((++prefableSpecs)->specs);

  return true;
}

bool
QueryInterface(JSContext* cx, unsigned argc, JS::Value* vp);

template <class T>
struct
WantsQueryInterface
{
  static_assert((IsBaseOf<nsISupports, T>::value),
                "QueryInterface can't work without an nsISupports.");
  static bool Enabled(JSContext* aCx, JSObject* aGlobal)
  {
    return NS_IsMainThread() && IsChromeOrXBL(aCx, aGlobal);
  }
};

bool
ThrowingConstructor(JSContext* cx, unsigned argc, JS::Value* vp);

// vp is allowed to be null; in that case no get will be attempted,
// and *found will simply indicate whether the property exists.
bool
GetPropertyOnPrototype(JSContext* cx, JS::Handle<JSObject*> proxy,
                       JS::Handle<jsid> id, bool* found,
                       JS::Value* vp);

bool
HasPropertyOnPrototype(JSContext* cx, JS::Handle<JSObject*> proxy,
                       JS::Handle<jsid> id);


// Append the property names in "names" to "props". If
// shadowPrototypeProperties is false then skip properties that are also
// present on the proto chain of proxy.  If shadowPrototypeProperties is true,
// then the "proxy" argument is ignored.
bool
AppendNamedPropertyIds(JSContext* cx, JS::Handle<JSObject*> proxy,
                       nsTArray<nsString>& names,
                       bool shadowPrototypeProperties, JS::AutoIdVector& props);

// A struct that has the same layout as an nsDependentString but much
// faster constructor and destructor behavior
struct FakeDependentString {
  FakeDependentString() :
    mFlags(nsDependentString::F_TERMINATED)
  {
  }

  void SetData(const nsDependentString::char_type* aData,
               nsDependentString::size_type aLength) {
    MOZ_ASSERT(mFlags == nsDependentString::F_TERMINATED);
    mData = aData;
    mLength = aLength;
  }

  void Truncate() {
    mData = nsDependentString::char_traits::sEmptyBuffer;
    mLength = 0;
  }

  void SetNull() {
    Truncate();
    mFlags |= nsDependentString::F_VOIDED;
  }

  const nsDependentString::char_type* Data() const
  {
    return mData;
  }

  nsDependentString::size_type Length() const
  {
    return mLength;
  }

  // If this ever changes, change the corresponding code in the
  // Optional<nsAString> specialization as well.
  const nsAString* ToAStringPtr() const {
    return reinterpret_cast<const nsDependentString*>(this);
  }

  nsAString* ToAStringPtr() {
    return reinterpret_cast<nsDependentString*>(this);
  }

  operator const nsAString& () const {
    return *reinterpret_cast<const nsDependentString*>(this);
  }

private:
  const nsDependentString::char_type* mData;
  nsDependentString::size_type mLength;
  uint32_t mFlags;

  // A class to use for our static asserts to ensure our object layout
  // matches that of nsDependentString.
  class DependentStringAsserter;
  friend class DependentStringAsserter;

  class DepedentStringAsserter : public nsDependentString {
  public:
    static void StaticAsserts() {
      static_assert(sizeof(FakeDependentString) == sizeof(nsDependentString),
                    "Must have right object size");
      static_assert((offsetof(FakeDependentString, mData) ==
                      offsetof(DepedentStringAsserter, mData)),
                    "Offset of mData should match");
      static_assert((offsetof(FakeDependentString, mLength) ==
                      offsetof(DepedentStringAsserter, mLength)),
                    "Offset of mLength should match");
      static_assert((offsetof(FakeDependentString, mFlags) ==
                      offsetof(DepedentStringAsserter, mFlags)),
                    "Offset of mFlags should match");
    }
  };
};

enum StringificationBehavior {
  eStringify,
  eEmpty,
  eNull
};

// pval must not be null and must point to a rooted JS::Value
static inline bool
ConvertJSValueToString(JSContext* cx, JS::Handle<JS::Value> v,
                       JS::MutableHandle<JS::Value> pval,
                       StringificationBehavior nullBehavior,
                       StringificationBehavior undefinedBehavior,
                       FakeDependentString& result)
{
  JSString *s;
  if (v.isString()) {
    s = v.toString();
  } else {
    StringificationBehavior behavior;
    if (v.isNull()) {
      behavior = nullBehavior;
    } else if (v.isUndefined()) {
      behavior = undefinedBehavior;
    } else {
      behavior = eStringify;
    }

    if (behavior != eStringify) {
      if (behavior == eEmpty) {
        result.Truncate();
      } else {
        result.SetNull();
      }
      return true;
    }

    s = JS_ValueToString(cx, v);
    if (!s) {
      return false;
    }
    pval.set(JS::StringValue(s));  // Root the new string.
  }

  size_t len;
  const jschar *chars = JS_GetStringCharsZAndLength(cx, s, &len);
  if (!chars) {
    return false;
  }

  result.SetData(chars, len);
  return true;
}

bool
ConvertJSValueToByteString(JSContext* cx, JS::Handle<JS::Value> v,
                           JS::MutableHandle<JS::Value> pval, bool nullable,
                           nsACString& result);

template<typename T>
void DoTraceSequence(JSTracer* trc, FallibleTArray<T>& seq);
template<typename T>
void DoTraceSequence(JSTracer* trc, InfallibleTArray<T>& seq);

// Class for simple sequence arguments, only used internally by codegen.
template<typename T>
class AutoSequence : public AutoFallibleTArray<T, 16>
{
public:
  AutoSequence() : AutoFallibleTArray<T, 16>()
  {}

  // Allow converting to const sequences as needed
  operator const Sequence<T>&() const {
    return *reinterpret_cast<const Sequence<T>*>(this);
  }
};

// Class used to trace sequences, with specializations for various
// sequence types.
template<typename T,
         bool isDictionary=IsBaseOf<DictionaryBase, T>::value,
         bool isTypedArray=IsBaseOf<AllTypedArraysBase, T>::value>
class SequenceTracer
{
  explicit SequenceTracer() MOZ_DELETE; // Should never be instantiated
};

// sequence<object> or sequence<object?>
template<>
class SequenceTracer<JSObject*, false, false>
{
  explicit SequenceTracer() MOZ_DELETE; // Should never be instantiated

public:
  static void TraceSequence(JSTracer* trc, JSObject** objp, JSObject** end) {
    for (; objp != end; ++objp) {
      JS_CallObjectTracer(trc, objp, "sequence<object>");
    }
  }
};

// sequence<any>
template<>
class SequenceTracer<JS::Value, false, false>
{
  explicit SequenceTracer() MOZ_DELETE; // Should never be instantiated

public:
  static void TraceSequence(JSTracer* trc, JS::Value* valp, JS::Value* end) {
    for (; valp != end; ++valp) {
      JS_CallValueTracer(trc, valp, "sequence<any>");
    }
  }
};

// sequence<sequence<T>>
template<typename T>
class SequenceTracer<Sequence<T>, false, false>
{
  explicit SequenceTracer() MOZ_DELETE; // Should never be instantiated

public:
  static void TraceSequence(JSTracer* trc, Sequence<T>* seqp, Sequence<T>* end) {
    for (; seqp != end; ++seqp) {
      DoTraceSequence(trc, *seqp);
    }
  }
};

// sequence<sequence<T>> as return value
template<typename T>
class SequenceTracer<nsTArray<T>, false, false>
{
  explicit SequenceTracer() MOZ_DELETE; // Should never be instantiated

public:
  static void TraceSequence(JSTracer* trc, nsTArray<T>* seqp, nsTArray<T>* end) {
    for (; seqp != end; ++seqp) {
      DoTraceSequence(trc, *seqp);
    }
  }
};

// sequence<someDictionary>
template<typename T>
class SequenceTracer<T, true, false>
{
  explicit SequenceTracer() MOZ_DELETE; // Should never be instantiated

public:
  static void TraceSequence(JSTracer* trc, T* dictp, T* end) {
    for (; dictp != end; ++dictp) {
      dictp->TraceDictionary(trc);
    }
  }
};

// sequence<SomeTypedArray>
template<typename T>
class SequenceTracer<T, false, true>
{
  explicit SequenceTracer() MOZ_DELETE; // Should never be instantiated

public:
  static void TraceSequence(JSTracer* trc, T* arrayp, T* end) {
    for (; arrayp != end; ++arrayp) {
      arrayp->TraceSelf(trc);
    }
  }
};

// sequence<T?> with T? being a Nullable<T>
template<typename T>
class SequenceTracer<Nullable<T>, false, false>
{
  explicit SequenceTracer() MOZ_DELETE; // Should never be instantiated

public:
  static void TraceSequence(JSTracer* trc, Nullable<T>* seqp,
                            Nullable<T>* end) {
    for (; seqp != end; ++seqp) {
      if (!seqp->IsNull()) {
        // Pretend like we actually have a length-one sequence here so
        // we can do template instantiation correctly for T.
        T& val = seqp->Value();
        T* ptr = &val;
        SequenceTracer<T>::TraceSequence(trc, ptr, ptr+1);
      }
    }
  }
};

template<typename T>
void DoTraceSequence(JSTracer* trc, FallibleTArray<T>& seq)
{
  SequenceTracer<T>::TraceSequence(trc, seq.Elements(),
                                   seq.Elements() + seq.Length());
}

template<typename T>
void DoTraceSequence(JSTracer* trc, InfallibleTArray<T>& seq)
{
  SequenceTracer<T>::TraceSequence(trc, seq.Elements(),
                                   seq.Elements() + seq.Length());
}

// Rooter class for sequences; this is what we mostly use in the codegen
template<typename T>
class MOZ_STACK_CLASS SequenceRooter : private JS::CustomAutoRooter
{
public:
  SequenceRooter(JSContext *aCx, FallibleTArray<T>* aSequence
                 MOZ_GUARD_OBJECT_NOTIFIER_PARAM)
    : JS::CustomAutoRooter(aCx MOZ_GUARD_OBJECT_NOTIFIER_PARAM_TO_PARENT),
      mFallibleArray(aSequence),
      mSequenceType(eFallibleArray)
  {
  }

  SequenceRooter(JSContext *aCx, InfallibleTArray<T>* aSequence
                 MOZ_GUARD_OBJECT_NOTIFIER_PARAM)
    : JS::CustomAutoRooter(aCx MOZ_GUARD_OBJECT_NOTIFIER_PARAM_TO_PARENT),
      mInfallibleArray(aSequence),
      mSequenceType(eInfallibleArray)
  {
  }

  SequenceRooter(JSContext *aCx, Nullable<nsTArray<T> >* aSequence
                 MOZ_GUARD_OBJECT_NOTIFIER_PARAM)
    : JS::CustomAutoRooter(aCx MOZ_GUARD_OBJECT_NOTIFIER_PARAM_TO_PARENT),
      mNullableArray(aSequence),
      mSequenceType(eNullableArray)
  {
  }

 private:
  enum SequenceType {
    eInfallibleArray,
    eFallibleArray,
    eNullableArray
  };

  virtual void trace(JSTracer *trc) MOZ_OVERRIDE
  {
    if (mSequenceType == eFallibleArray) {
      DoTraceSequence(trc, *mFallibleArray);
    } else if (mSequenceType == eInfallibleArray) {
      DoTraceSequence(trc, *mInfallibleArray);
    } else {
      MOZ_ASSERT(mSequenceType == eNullableArray);
      if (!mNullableArray->IsNull()) {
        DoTraceSequence(trc, mNullableArray->Value());
      }
    }
  }

  union {
    InfallibleTArray<T>* mInfallibleArray;
    FallibleTArray<T>* mFallibleArray;
    Nullable<nsTArray<T> >* mNullableArray;
  };

  SequenceType mSequenceType;
};

template<typename T>
class MOZ_STACK_CLASS RootedUnion : public T,
                                    private JS::CustomAutoRooter
{
public:
  RootedUnion(JSContext* cx MOZ_GUARD_OBJECT_NOTIFIER_PARAM) :
    T(),
    JS::CustomAutoRooter(cx MOZ_GUARD_OBJECT_NOTIFIER_PARAM_TO_PARENT)
  {
  }

  virtual void trace(JSTracer *trc) MOZ_OVERRIDE
  {
    this->TraceUnion(trc);
  }
};

template<typename T>
class MOZ_STACK_CLASS NullableRootedUnion : public Nullable<T>,
                                            private JS::CustomAutoRooter
{
public:
  NullableRootedUnion(JSContext* cx MOZ_GUARD_OBJECT_NOTIFIER_PARAM) :
    Nullable<T>(),
    JS::CustomAutoRooter(cx MOZ_GUARD_OBJECT_NOTIFIER_PARAM_TO_PARENT)
  {
  }

  virtual void trace(JSTracer *trc) MOZ_OVERRIDE
  {
    if (!this->IsNull()) {
      this->Value().TraceUnion(trc);
    }
  }
};

inline bool
IdEquals(jsid id, const char* string)
{
  return JSID_IS_STRING(id) &&
         JS_FlatStringEqualsAscii(JSID_TO_FLAT_STRING(id), string);
}

inline bool
AddStringToIDVector(JSContext* cx, JS::AutoIdVector& vector, const char* name)
{
  return vector.growBy(1) &&
         InternJSString(cx, vector[vector.length() - 1], name);
}

// Implementation of the bits that XrayWrapper needs

/**
 * This resolves indexed or named properties of obj.
 *
 * wrapper is the Xray JS object.
 * obj is the target object of the Xray, a binding's instance object or a
 *     interface or interface prototype object.
 */
bool
XrayResolveOwnProperty(JSContext* cx, JS::Handle<JSObject*> wrapper,
                       JS::Handle<JSObject*> obj,
                       JS::Handle<jsid> id,
                       JS::MutableHandle<JSPropertyDescriptor> desc, unsigned flags);

/**
 * This resolves operations, attributes and constants of the interfaces for obj.
 *
 * wrapper is the Xray JS object.
 * obj is the target object of the Xray, a binding's instance object or a
 *     interface or interface prototype object.
 */
bool
XrayResolveNativeProperty(JSContext* cx, JS::Handle<JSObject*> wrapper,
                          JS::Handle<JSObject*> obj,
                          JS::Handle<jsid> id, JS::MutableHandle<JSPropertyDescriptor> desc);

/**
 * Define a property on obj through an Xray wrapper.
 *
 * wrapper is the Xray JS object.
 * obj is the target object of the Xray, a binding's instance object or a
 *     interface or interface prototype object.
 * defined will be set to true if a property was set as a result of this call.
 */
bool
XrayDefineProperty(JSContext* cx, JS::Handle<JSObject*> wrapper,
                   JS::Handle<JSObject*> obj, JS::Handle<jsid> id,
                   JS::MutableHandle<JSPropertyDescriptor> desc, bool* defined);

/**
 * This enumerates indexed or named properties of obj and operations, attributes
 * and constants of the interfaces for obj.
 *
 * wrapper is the Xray JS object.
 * obj is the target object of the Xray, a binding's instance object or a
 *     interface or interface prototype object.
 * flags are JSITER_* flags.
 */
bool
XrayEnumerateProperties(JSContext* cx, JS::Handle<JSObject*> wrapper,
                        JS::Handle<JSObject*> obj,
                        unsigned flags, JS::AutoIdVector& props);

extern NativePropertyHooks sWorkerNativePropertyHooks;

// We use one constructor JSNative to represent all DOM interface objects (so
// we can easily detect when we need to wrap them in an Xray wrapper). We store
// the real JSNative in the mNative member of a JSNativeHolder in the
// CONSTRUCTOR_NATIVE_HOLDER_RESERVED_SLOT slot of the JSFunction object for a
// specific interface object. We also store the NativeProperties in the
// JSNativeHolder. The CONSTRUCTOR_XRAY_EXPANDO_SLOT is used to store the
// expando chain of the Xray for the interface object.
// Note that some interface objects are not yet a JSFunction but a normal
// JSObject with a DOMJSClass, those do not use these slots.

enum {
  CONSTRUCTOR_NATIVE_HOLDER_RESERVED_SLOT = 0,
  CONSTRUCTOR_XRAY_EXPANDO_SLOT
};

bool
Constructor(JSContext* cx, unsigned argc, JS::Value* vp);

inline bool
UseDOMXray(JSObject* obj)
{
  const js::Class* clasp = js::GetObjectClass(obj);
  return IsDOMClass(clasp) ||
         IsDOMProxy(obj, clasp) ||
         JS_IsNativeFunction(obj, Constructor) ||
         IsDOMIfaceAndProtoClass(clasp);
}

#ifdef DEBUG
inline bool
HasConstructor(JSObject* obj)
{
  return JS_IsNativeFunction(obj, Constructor) ||
         js::GetObjectClass(obj)->construct;
}
 #endif
 
// Transfer reference in ptr to smartPtr.
template<class T>
inline void
Take(nsRefPtr<T>& smartPtr, T* ptr)
{
  smartPtr = dont_AddRef(ptr);
}

// Transfer ownership of ptr to smartPtr.
template<class T>
inline void
Take(nsAutoPtr<T>& smartPtr, T* ptr)
{
  smartPtr = ptr;
}

inline void
MustInheritFromNonRefcountedDOMObject(NonRefcountedDOMObject*)
{
}

// Set the chain of expando objects for various consumers of the given object.
// For Paris Bindings only. See the relevant infrastructure in XrayWrapper.cpp.
JSObject* GetXrayExpandoChain(JSObject *obj);
void SetXrayExpandoChain(JSObject *obj, JSObject *chain);

/**
 * This creates a JSString containing the value that the toString function for
 * obj should create according to the WebIDL specification, ignoring any
 * modifications by script. The value is prefixed with pre and postfixed with
 * post, unless this is called for an object that has a stringifier. It is
 * specifically for use by Xray code.
 *
 * wrapper is the Xray JS object.
 * obj is the target object of the Xray, a binding's instance object or a
 *     interface or interface prototype object.
 * pre is a string that should be prefixed to the value.
 * post is a string that should be prefixed to the value.
 * v contains the JSString for the value if the function returns true.
 */
bool
NativeToString(JSContext* cx, JS::Handle<JSObject*> wrapper,
               JS::Handle<JSObject*> obj, const char* pre,
               const char* post,
               JS::MutableHandle<JS::Value> v);

HAS_MEMBER(JSBindingFinalized)

template<class T, bool hasCallback=HasJSBindingFinalizedMember<T>::Value>
struct JSBindingFinalized
{
  static void Finalized(T* self)
  {
  }
};

template<class T>
struct JSBindingFinalized<T, true>
{
  static void Finalized(T* self)
  {
    self->JSBindingFinalized();
  }
};

// Helpers for creating a const version of a type.
template<typename T>
const T& Constify(T& arg)
{
  return arg;
}

// Helper for turning (Owning)NonNull<T> into T&
template<typename T>
T& NonNullHelper(T& aArg)
{
  return aArg;
}

template<typename T>
T& NonNullHelper(NonNull<T>& aArg)
{
  return aArg;
}

template<typename T>
const T& NonNullHelper(const NonNull<T>& aArg)
{
  return aArg;
}

template<typename T>
T& NonNullHelper(OwningNonNull<T>& aArg)
{
  return aArg;
}

template<typename T>
const T& NonNullHelper(const OwningNonNull<T>& aArg)
{
  return aArg;
}

// Reparent the wrapper of aObj to whatever its native now thinks its
// parent should be.
nsresult
ReparentWrapper(JSContext* aCx, JS::Handle<JSObject*> aObj);

/**
 * Used to implement the hasInstance hook of an interface object.
 *
 * instance should not be a security wrapper.
 */
bool
InterfaceHasInstance(JSContext* cx, JS::Handle<JSObject*> obj,
                     JS::Handle<JSObject*> instance,
                     bool* bp);
bool
InterfaceHasInstance(JSContext* cx, JS::Handle<JSObject*> obj, JS::MutableHandle<JS::Value> vp,
                     bool* bp);
bool
InterfaceHasInstance(JSContext* cx, int prototypeID, int depth,
                     JS::Handle<JSObject*> instance,
                     bool* bp);

// Helper for lenient getters/setters to report to console.  If this
// returns false, we couldn't even get a global.
bool
ReportLenientThisUnwrappingFailure(JSContext* cx, JSObject* obj);

inline JSObject*
GetUnforgeableHolder(JSObject* aGlobal, prototypes::ID aId)
{
  JS::Heap<JSObject*>* protoAndIfaceArray = GetProtoAndIfaceArray(aGlobal);
  JSObject* interfaceProto = protoAndIfaceArray[aId];
  return &js::GetReservedSlot(interfaceProto,
                              DOM_INTERFACE_PROTO_SLOTS_BASE).toObject();
}

// Given a JSObject* that represents the chrome side of a JS-implemented WebIDL
// interface, get the nsPIDOMWindow corresponding to the content side, if any.
// A false return means an exception was thrown.
bool
GetWindowForJSImplementedObject(JSContext* cx, JS::Handle<JSObject*> obj,
                                nsPIDOMWindow** window);

already_AddRefed<nsPIDOMWindow>
ConstructJSImplementation(JSContext* aCx, const char* aContractId,
                          const GlobalObject& aGlobal,
                          JS::MutableHandle<JSObject*> aObject,
                          ErrorResult& aRv);

/**
 * Convert an nsCString to jsval, returning true on success.
 * These functions are intended for ByteString implementations.
 * As such, the string is not UTF-8 encoded.  Any UTF8 strings passed to these
 * methods will be mangled.
 */
bool NonVoidByteStringToJsval(JSContext *cx, const nsACString &str,
                              JS::MutableHandle<JS::Value> rval);
inline bool ByteStringToJsval(JSContext *cx, const nsACString &str,
                              JS::MutableHandle<JS::Value> rval)
{
    if (str.IsVoid()) {
        rval.setNull();
        return true;
    }
    return NonVoidByteStringToJsval(cx, str, rval);
}

template<class T, bool isISupports=IsBaseOf<nsISupports, T>::value>
struct PreserveWrapperHelper
{
  static void PreserveWrapper(T* aObject)
  {
    aObject->PreserveWrapper(aObject, NS_CYCLE_COLLECTION_PARTICIPANT(T));
  }
};

template<class T>
struct PreserveWrapperHelper<T, true>
{
  static void PreserveWrapper(T* aObject)
  {
    aObject->PreserveWrapper(reinterpret_cast<nsISupports*>(aObject));
  }
};

template<class T>
void PreserveWrapper(T* aObject)
{
  PreserveWrapperHelper<T>::PreserveWrapper(aObject);
}

template<class T, bool isISupports=IsBaseOf<nsISupports, T>::value>
struct CastingAssertions
{
  static bool ToSupportsIsCorrect(T*)
  {
    return true;
  }
  static bool ToSupportsIsOnPrimaryInheritanceChain(T*, nsWrapperCache*)
  {
    return true;
  }
};

template<class T>
struct CastingAssertions<T, true>
{
  static bool ToSupportsIsCorrect(T* aObject)
  {
    return ToSupports(aObject) ==  reinterpret_cast<nsISupports*>(aObject);
  }
  static bool ToSupportsIsOnPrimaryInheritanceChain(T* aObject,
                                                    nsWrapperCache* aCache)
  {
    return reinterpret_cast<void*>(aObject) != aCache;
  }
};

template<class T>
bool
ToSupportsIsCorrect(T* aObject)
{
  return CastingAssertions<T>::ToSupportsIsCorrect(aObject);
}

template<class T>
bool
ToSupportsIsOnPrimaryInheritanceChain(T* aObject, nsWrapperCache* aCache)
{
  return CastingAssertions<T>::ToSupportsIsOnPrimaryInheritanceChain(aObject,
                                                                     aCache);
}

template<class T, template <typename> class SmartPtr,
         bool isISupports=IsBaseOf<nsISupports, T>::value>
class DeferredFinalizer
{
  typedef nsTArray<SmartPtr<T> > SmartPtrArray;

  static void*
  AppendDeferredFinalizePointer(void* aData, void* aObject)
  {
    SmartPtrArray* pointers = static_cast<SmartPtrArray*>(aData);
    if (!pointers) {
      pointers = new SmartPtrArray();
    }

    T* self = static_cast<T*>(aObject);

    SmartPtr<T>* defer = pointers->AppendElement();
    Take(*defer, self);
    return pointers;
  }
  static bool
  DeferredFinalize(uint32_t aSlice, void* aData)
  {
    MOZ_ASSERT(aSlice > 0, "nonsensical/useless call with aSlice == 0");
    SmartPtrArray* pointers = static_cast<SmartPtrArray*>(aData);
    uint32_t oldLen = pointers->Length();
    if (oldLen < aSlice) {
      aSlice = oldLen;
    }
    uint32_t newLen = oldLen - aSlice;
    pointers->RemoveElementsAt(newLen, aSlice);
    if (newLen == 0) {
      delete pointers;
      return true;
    }
    return false;
  }

public:
  static void
  AddForDeferredFinalization(T* aObject)
  {
    cyclecollector::DeferredFinalize(AppendDeferredFinalizePointer,
                                     DeferredFinalize, aObject);
  }
};

template<class T, template <typename> class SmartPtr>
class DeferredFinalizer<T, SmartPtr, true>
{
public:
  static void
  AddForDeferredFinalization(T* aObject)
  {
    cyclecollector::DeferredFinalize(reinterpret_cast<nsISupports*>(aObject));
  }
};

template<class T, template <typename> class SmartPtr>
static void
AddForDeferredFinalization(T* aObject)
{
  DeferredFinalizer<T, SmartPtr>::AddForDeferredFinalization(aObject);
}

// This returns T's CC participant if it participates in CC or null if it
// doesn't. This also returns null for classes that don't inherit from
// nsISupports (QI should be used to get the participant for those).
template<class T, bool isISupports=IsBaseOf<nsISupports, T>::value>
class GetCCParticipant
{
  // Helper for GetCCParticipant for classes that participate in CC.
  template<class U>
  static MOZ_CONSTEXPR nsCycleCollectionParticipant*
  GetHelper(int, typename U::NS_CYCLE_COLLECTION_INNERCLASS* dummy=nullptr)
  {
    return T::NS_CYCLE_COLLECTION_INNERCLASS::GetParticipant();
  }
  // Helper for GetCCParticipant for classes that don't participate in CC.
  template<class U>
  static MOZ_CONSTEXPR nsCycleCollectionParticipant*
  GetHelper(double)
  {
    return nullptr;
  }

public:
  static MOZ_CONSTEXPR nsCycleCollectionParticipant*
  Get()
  {
    // Passing int() here will try to call the GetHelper that takes an int as
    // its firt argument. If T doesn't participate in CC then substitution for
    // the second argument (with a default value) will fail and because of
    // SFINAE the next best match (the variant taking a double) will be called.
    return GetHelper<T>(int());
  }
};

template<class T>
class GetCCParticipant<T, true>
{
public:
  static MOZ_CONSTEXPR nsCycleCollectionParticipant*
  Get()
  {
    return nullptr;
  }
};

bool
ThreadsafeCheckIsChrome(JSContext* aCx, JSObject* aObj);

} // namespace dom
} // namespace mozilla

#endif /* mozilla_dom_BindingUtils_h__ */
