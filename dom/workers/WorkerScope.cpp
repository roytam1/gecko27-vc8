/* -*- Mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; tab-width: 40 -*- */
/* vim: set ts=2 et sw=2 tw=80: */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#if defined(_MSC_VER) && _MSC_VER < 1500
#include <boost/typeof/typeof.hpp>
#endif

#include "WorkerScope.h"

#include "jsapi.h"
#include "js/OldDebugAPI.h"
#include "mozilla/Util.h"
#include "mozilla/dom/DOMJSClass.h"
#include "mozilla/dom/EventBinding.h"
#include "mozilla/dom/EventHandlerBinding.h"
#include "mozilla/dom/EventTargetBinding.h"
#include "mozilla/dom/BindingUtils.h"
#include "mozilla/dom/DOMExceptionBinding.h"
#include "mozilla/dom/FileReaderSyncBinding.h"
#include "mozilla/dom/ImageData.h"
#include "mozilla/dom/ImageDataBinding.h"
#include "mozilla/dom/TextDecoderBinding.h"
#include "mozilla/dom/TextEncoderBinding.h"
#include "mozilla/dom/XMLHttpRequestBinding.h"
#include "mozilla/dom/XMLHttpRequestUploadBinding.h"
#include "mozilla/dom/URLBinding.h"
#include "mozilla/dom/WorkerLocationBinding.h"
#include "mozilla/dom/WorkerNavigatorBinding.h"
#include "mozilla/OSFileConstants.h"
#include "nsIGlobalObject.h"
#include "nsTraceRefcnt.h"
#include "xpcpublic.h"

#ifdef ANDROID
#include <android/log.h>
#endif

#include "ChromeWorkerScope.h"
#include "Events.h"
#include "EventListenerManager.h"
#include "EventTarget.h"
#include "File.h"
#include "FileReaderSync.h"
#include "Location.h"
#include "Navigator.h"
#include "Principal.h"
#include "ScriptLoader.h"
#include "Worker.h"
#include "WorkerPrivate.h"
#include "XMLHttpRequest.h"

#include "WorkerInlines.h"

#define FUNCTION_FLAGS \
  JSPROP_ENUMERATE

using namespace mozilla;
using namespace mozilla::dom;
USING_WORKERS_NAMESPACE

namespace {

class WorkerGlobalScope : public workers::EventTarget,
                          public nsIGlobalObject
{
  static const JSClass sClass;
  static const JSPropertySpec sProperties[];
  static const JSFunctionSpec sFunctions[];

  enum
  {
    SLOT_wrappedScope = 0,
    SLOT_wrappedFunction
  };

  enum
  {
    SLOT_location = 0,
    SLOT_navigator,

    SLOT_COUNT
  };

  // Must be traced!
  JS::Heap<JS::Value> mSlots[SLOT_COUNT];

  enum
  {
    STRING_onerror = 0,
    STRING_onclose,

    STRING_COUNT
  };

  static const char* const sEventStrings[STRING_COUNT];

protected:
  WorkerPrivate* mWorker;

public:
  static const JSClass*
  Class()
  {
    return &sClass;
  }

  static JSObject*
  InitClass(JSContext* aCx, JSObject* aObj, JSObject* aParentProto)
  {
    return JS_InitClass(aCx, aObj, aParentProto, Class(), Construct, 0,
                        sProperties, sFunctions, NULL, NULL);
  }

  using EventTarget::GetEventListener;
  using EventTarget::SetEventListener;

protected:
  WorkerGlobalScope(JSContext* aCx, WorkerPrivate* aWorker)
  : EventTarget(aCx), mWorker(aWorker)
  {
    MOZ_COUNT_CTOR(mozilla::dom::workers::WorkerGlobalScope);
    for (int32_t i = 0; i < SLOT_COUNT; i++) {
      mSlots[i] = JSVAL_VOID;
    }
  }

  ~WorkerGlobalScope()
  {
    MOZ_COUNT_DTOR(mozilla::dom::workers::WorkerGlobalScope);
  }

  NS_DECL_ISUPPORTS_INHERITED

  // nsIGlobalObject
  virtual JSObject* GetGlobalJSObject() MOZ_OVERRIDE
  {
    mWorker->AssertIsOnWorkerThread();
    return GetJSObject();
  }

  virtual void
  _trace(JSTracer* aTrc) MOZ_OVERRIDE
  {
    for (int32_t i = 0; i < SLOT_COUNT; i++) {
      JS_CallHeapValueTracer(aTrc, &mSlots[i],
                             "WorkerGlobalScope instance slot");
    }
    mWorker->TraceInternal(aTrc);
    EventTarget::_trace(aTrc);
  }

  virtual void
  _finalize(JSFreeOp* aFop) MOZ_OVERRIDE
  {
    EventTarget::_finalize(aFop);
  }

private:
  static bool IsWorkerGlobalScope(JS::Handle<JS::Value> v);

  static bool
  GetOnCloseImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
    const char* name = sEventStrings[STRING_onclose];
    WorkerGlobalScope* scope = GetInstancePrivate(aCx, &aArgs.thisv().toObject(), name);
    MOZ_ASSERT(scope);

    ErrorResult rv;
    nsRefPtr<EventHandlerNonNull> handler =
      scope->GetEventListener(NS_ConvertASCIItoUTF16(name + 2), rv);

    if (rv.Failed()) {
      JS_ReportError(aCx, "Failed to get event listener!");
      return false;
    }

    if (!handler) {
      aArgs.rval().setNull();
    } else {
      aArgs.rval().setObject(*handler->Callable());
    }
    return true;
  }

  static bool
  GetOnClose(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
    JS::CallArgs args = JS::CallArgsFromVp(aArgc, aVp);
    return JS::CallNonGenericMethod<IsWorkerGlobalScope, GetOnCloseImpl>(aCx, args);
  }

  static bool
  SetOnCloseImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
    const char* name = sEventStrings[STRING_onclose];
    WorkerGlobalScope* scope =
      GetInstancePrivate(aCx, &aArgs.thisv().toObject(), name);
    MOZ_ASSERT(scope);

    if (aArgs.length() == 0 || !aArgs[0].isObjectOrNull()) {
      JS_ReportError(aCx, "Not an event listener!");
      return false;
    }

    ErrorResult rv;
    JS::Rooted<JSObject*> listenerObj(aCx, aArgs[0].toObjectOrNull());
    nsRefPtr<EventHandlerNonNull> handler;
    if (listenerObj && JS_ObjectIsCallable(aCx, listenerObj)) {
      handler = new EventHandlerNonNull(listenerObj);
    } else {
      handler = nullptr;
    }
    scope->SetEventListener(NS_ConvertASCIItoUTF16(name + 2),
                            handler, rv);
    if (rv.Failed()) {
      JS_ReportError(aCx, "Failed to set event listener!");
      return false;
    }

    aArgs.rval().setUndefined();
    return true;
  }

  static bool
  SetOnClose(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
    JS::CallArgs args = JS::CallArgsFromVp(aArgc, aVp);
    return JS::CallNonGenericMethod<IsWorkerGlobalScope, SetOnCloseImpl>(aCx, args);
  }

  static WorkerGlobalScope*
  GetInstancePrivate(JSContext* aCx, JSObject* aObj, const char* aFunctionName);

  static bool
  Construct(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS_ReportErrorNumber(aCx, js_GetErrorMessage, NULL, JSMSG_WRONG_CONSTRUCTOR,
                         sClass.name);
    return false;
  }

  static bool
  GetSelfImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
    aArgs.rval().setObject(aArgs.thisv().toObject());
    return true;
  }

  static bool
  GetSelf(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
    JS::CallArgs args = JS::CallArgsFromVp(aArgc, aVp);
    return JS::CallNonGenericMethod<IsWorkerGlobalScope, GetSelfImpl>(aCx, args);
  }

  static bool
  GetLocationImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
    JS::Rooted<JSObject*> obj(aCx, &aArgs.thisv().toObject());
    WorkerGlobalScope* scope =
      GetInstancePrivate(aCx, obj, sProperties[SLOT_location].name);
    MOZ_ASSERT(scope);

    if (scope->mSlots[SLOT_location].isUndefined()) {
      WorkerPrivate::LocationInfo& info = scope->mWorker->GetLocationInfo();

      nsRefPtr<WorkerLocation> location =
        WorkerLocation::Create(aCx, obj, info);
      if (!location) {
        return false;
      }

      JS::Rooted<JS::Value> val(aCx);
      if (!WrapNewBindingObject(aCx, obj, location, &val)) {
        return false;
      }

      scope->mSlots[SLOT_location] = val;
    }

    aArgs.rval().set(scope->mSlots[SLOT_location]);
    return true;
  }

  static bool
  GetLocation(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
    JS::CallArgs args = JS::CallArgsFromVp(aArgc, aVp);
    return JS::CallNonGenericMethod<IsWorkerGlobalScope, GetLocationImpl>(aCx, args);
  }

  static bool
  UnwrapErrorEvent(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS_ASSERT(aArgc == 1);
    JS_ASSERT((JS_ARGV(aCx, aVp)[0]).isObject());

    JSObject* wrapper = &JS_CALLEE(aCx, aVp).toObject();
    JS_ASSERT(JS_ObjectIsFunction(aCx, wrapper));

    JS::Rooted<JS::Value> scope(aCx,
      js::GetFunctionNativeReserved(wrapper, SLOT_wrappedScope));
    JS::Rooted<JS::Value> listener(aCx,
      js::GetFunctionNativeReserved(wrapper, SLOT_wrappedFunction));

    JS_ASSERT(scope.isObject());

    JS::Rooted<JSObject*> event(aCx, &JS_ARGV(aCx, aVp)[0].toObject());

    jsval argv[3] = { JSVAL_VOID, JSVAL_VOID, JSVAL_VOID };
    JS::AutoArrayRooter rootedArgv(aCx, ArrayLength(argv), argv);
    if (!JS_GetProperty(aCx, event, "message", rootedArgv.handleAt(0)) ||
        !JS_GetProperty(aCx, event, "filename", rootedArgv.handleAt(1)) ||
        !JS_GetProperty(aCx, event, "lineno", rootedArgv.handleAt(2))) {
      return false;
    }

    JS::Rooted<JS::Value> rval(aCx, JS::UndefinedValue());
    if (!JS_CallFunctionValue(aCx, JSVAL_TO_OBJECT(scope), listener,
                              ArrayLength(argv), argv, rval.address())) {
      JS_ReportPendingException(aCx);
      return false;
    }

    if (JSVAL_IS_BOOLEAN(rval) && JSVAL_TO_BOOLEAN(rval) &&
        !JS_CallFunctionName(aCx, event, "preventDefault", 0, NULL,
                             rval.address())) {
      return false;
    }

    return true;
  }

  static bool
  GetOnErrorListenerImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
    const char* name = sEventStrings[STRING_onerror];
    WorkerGlobalScope* scope = GetInstancePrivate(aCx, &aArgs.thisv().toObject(), name);
    MOZ_ASSERT(scope);

    ErrorResult rv;

    nsRefPtr<EventHandlerNonNull> adaptor =
      scope->GetEventListener(NS_ConvertASCIItoUTF16(name + 2), rv);
    if (rv.Failed()) {
      JS_ReportError(aCx, "Failed to get event listener!");
      return false;
    }

    if (!adaptor) {
      aArgs.rval().setNull();
      return true;
    }

    aArgs.rval().set(js::GetFunctionNativeReserved(adaptor->Callable(),
                                                   SLOT_wrappedFunction));
    MOZ_ASSERT(aArgs.rval().isObject());
    return true;
  }

  static bool
  GetOnErrorListener(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
    JS::CallArgs args = JS::CallArgsFromVp(aArgc, aVp);
    return JS::CallNonGenericMethod<IsWorkerGlobalScope, GetOnErrorListenerImpl>(aCx, args);
  }

  static bool
  SetOnErrorListenerImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
    JS::Rooted<JSObject*> obj(aCx, &aArgs.thisv().toObject());
    const char* name = sEventStrings[STRING_onerror];
    WorkerGlobalScope* scope = GetInstancePrivate(aCx, obj, name);
    MOZ_ASSERT(scope);

    if (aArgs.length() == 0 || !aArgs[0].isObject()) {
      JS_ReportError(aCx, "Not an event listener!");
      return false;
    }

    JSFunction* adaptor =
      js::NewFunctionWithReserved(aCx, UnwrapErrorEvent, 1, 0,
                                  JS::CurrentGlobalOrNull(aCx), "unwrap");
    if (!adaptor) {
      return false;
    }

    JS::Rooted<JSObject*> listener(aCx, JS_GetFunctionObject(adaptor));
    if (!listener) {
      return false;
    }

    js::SetFunctionNativeReserved(listener, SLOT_wrappedScope,
                                  JS::ObjectValue(*obj));
    js::SetFunctionNativeReserved(listener, SLOT_wrappedFunction, aArgs[0]);

    ErrorResult rv;
    nsRefPtr<EventHandlerNonNull> handler = new EventHandlerNonNull(listener);
    scope->SetEventListener(NS_ConvertASCIItoUTF16(name + 2), handler, rv);

    if (rv.Failed()) {
      JS_ReportError(aCx, "Failed to set event listener!");
      return false;
    }

    aArgs.rval().setUndefined();
    return true;
  }

  static bool
  SetOnErrorListener(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
    JS::CallArgs args = JS::CallArgsFromVp(aArgc, aVp);
    return JS::CallNonGenericMethod<IsWorkerGlobalScope, SetOnErrorListenerImpl>(aCx, args);
  }

  static bool
  GetNavigatorImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
    JS::Rooted<JSObject*> obj(aCx, &aArgs.thisv().toObject());
    WorkerGlobalScope* scope =
      GetInstancePrivate(aCx, obj, sProperties[SLOT_navigator].name);
    MOZ_ASSERT(scope);

    if (scope->mSlots[SLOT_navigator].isUndefined()) {
      nsRefPtr<WorkerNavigator> navigator = WorkerNavigator::Create(aCx, obj);
      if (!navigator) {
        return false;
      }

      JS::Rooted<JS::Value> val(aCx);
      if (!WrapNewBindingObject(aCx, obj, navigator, &val)) {
        return false;
      }

      scope->mSlots[SLOT_navigator] = val;
    }

    aArgs.rval().set(scope->mSlots[SLOT_navigator]);
    return true;
  }

  static bool
  GetNavigator(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
    JS::CallArgs args = JS::CallArgsFromVp(aArgc, aVp);
    return JS::CallNonGenericMethod<IsWorkerGlobalScope, GetNavigatorImpl>(aCx, args);
  }

  static bool
  Close(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    WorkerGlobalScope* scope = GetInstancePrivate(aCx, obj, sFunctions[0].name);
    if (!scope) {
      return false;
    }

    if (!scope->mWorker->CloseInternal(aCx)) {
      return false;
    }

    JS_RVAL(aCx, aVp).setUndefined();
    return true;
  }

  static bool
  ImportScripts(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    WorkerGlobalScope* scope = GetInstancePrivate(aCx, obj, sFunctions[1].name);
    if (!scope) {
      return false;
    }

    if (aArgc && !scriptloader::Load(aCx, aArgc, JS_ARGV(aCx, aVp))) {
      return false;
    }

    JS_RVAL(aCx, aVp).setUndefined();
    return true;
  }

  static bool
  SetTimeout(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    WorkerGlobalScope* scope = GetInstancePrivate(aCx, obj, sFunctions[2].name);
    if (!scope) {
      return false;
    }

    JS::Rooted<JS::Value> dummy(aCx);
    if (!JS_ConvertArguments(aCx, aArgc, JS_ARGV(aCx, aVp), "v",
                             dummy.address())) {
      return false;
    }

    return scope->mWorker->SetTimeout(aCx, aArgc, aVp, false);
  }

  static bool
  ClearTimeout(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    WorkerGlobalScope* scope = GetInstancePrivate(aCx, obj, sFunctions[3].name);
    if (!scope) {
      return false;
    }

    uint32_t id;
    if (!JS_ConvertArguments(aCx, aArgc, JS_ARGV(aCx, aVp), "u", &id)) {
      return false;
    }

    if (!scope->mWorker->ClearTimeout(aCx, id)) {
      return false;
    }

    JS_RVAL(aCx, aVp).setUndefined();
    return true;
  }

  static bool
  SetInterval(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    WorkerGlobalScope* scope = GetInstancePrivate(aCx, obj, sFunctions[4].name);
    if (!scope) {
      return false;
    }

    JS::Rooted<JS::Value> dummy(aCx);
    if (!JS_ConvertArguments(aCx, aArgc, JS_ARGV(aCx, aVp), "v",
                             dummy.address())) {
      return false;
    }

    return scope->mWorker->SetTimeout(aCx, aArgc, aVp, true);
  }

  static bool
  ClearInterval(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    WorkerGlobalScope* scope = GetInstancePrivate(aCx, obj, sFunctions[5].name);
    if (!scope) {
      return false;
    }

    uint32_t id;
    if (!JS_ConvertArguments(aCx, aArgc, JS_ARGV(aCx, aVp), "u", &id)) {
      return false;
    }

    if (!scope->mWorker->ClearTimeout(aCx, id)) {
      return false;
    }

    JS_RVAL(aCx, aVp).setUndefined();
    return true;
  }

  static bool
  Dump(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    if (!GetInstancePrivate(aCx, obj, sFunctions[6].name)) {
      return false;
    }

    if (aArgc) {
      JSString* str = JS_ValueToString(aCx, JS_ARGV(aCx, aVp)[0]);
      if (!str) {
        return false;
      }

      JSAutoByteString buffer(aCx, str);
      if (!buffer) {
        return false;
      }

#ifdef ANDROID
      __android_log_print(ANDROID_LOG_INFO, "Gecko", "%s", buffer.ptr());
#endif
      fputs(buffer.ptr(), stdout);
      fflush(stdout);
    }

    JS_RVAL(aCx, aVp).setUndefined();
    return true;
  }

  static bool
  AtoB(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    if (!GetInstancePrivate(aCx, obj, sFunctions[7].name)) {
      return false;
    }

    JS::Rooted<JS::Value> string(aCx);
    if (!JS_ConvertArguments(aCx, aArgc, JS_ARGV(aCx, aVp), "v",
                             string.address())) {
      return false;
    }

    JS::Rooted<JS::Value> result(aCx);
    if (!xpc::Base64Decode(aCx, string, result.address())) {
      return false;
    }

    JS_SET_RVAL(aCx, aVp, result);
    return true;
  }

  static bool
  BtoA(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    if (!GetInstancePrivate(aCx, obj, sFunctions[8].name)) {
      return false;
    }

    JS::Rooted<JS::Value> binary(aCx);
    if (!JS_ConvertArguments(aCx, aArgc, JS_ARGV(aCx, aVp), "v",
                             binary.address())) {
      return false;
    }

    JS::Rooted<JS::Value> result(aCx);
    if (!xpc::Base64Encode(aCx, binary, result.address())) {
      return false;
    }

    JS_SET_RVAL(aCx, aVp, result);
    return true;
  }
};

NS_IMPL_ADDREF_INHERITED(WorkerGlobalScope, workers::EventTarget)
NS_IMPL_RELEASE_INHERITED(WorkerGlobalScope, workers::EventTarget)

NS_INTERFACE_MAP_BEGIN(WorkerGlobalScope)
  NS_INTERFACE_MAP_ENTRY(nsIGlobalObject)
  NS_INTERFACE_MAP_ENTRY_AMBIGUOUS(nsISupports, DOMBindingBase)
NS_INTERFACE_MAP_END

const JSClass WorkerGlobalScope::sClass = {
  "WorkerGlobalScope",
  0,
  JS_PropertyStub, JS_DeletePropertyStub, JS_PropertyStub,
  JS_StrictPropertyStub, JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub
};

const JSPropertySpec WorkerGlobalScope::sProperties[] = {
  JS_PSGS("location", GetLocation, GetterOnlyJSNative, JSPROP_ENUMERATE),
  JS_PSGS(sEventStrings[STRING_onerror], GetOnErrorListener, SetOnErrorListener,
          JSPROP_ENUMERATE),
  JS_PSGS(sEventStrings[STRING_onclose], GetOnClose, SetOnClose,
          JSPROP_ENUMERATE),
  JS_PSGS("navigator", GetNavigator, GetterOnlyJSNative, JSPROP_ENUMERATE),
  JS_PSGS("self", GetSelf, GetterOnlyJSNative, JSPROP_ENUMERATE),
  JS_PS_END
};

const JSFunctionSpec WorkerGlobalScope::sFunctions[] = {
  JS_FN("close", Close, 0, FUNCTION_FLAGS),
  JS_FN("importScripts", ImportScripts, 1, FUNCTION_FLAGS),
  JS_FN("setTimeout", SetTimeout, 1, FUNCTION_FLAGS),
  JS_FN("clearTimeout", ClearTimeout, 1, FUNCTION_FLAGS),
  JS_FN("setInterval", SetInterval, 1, FUNCTION_FLAGS),
  JS_FN("clearInterval", ClearTimeout, 1, FUNCTION_FLAGS),
  JS_FN("dump", Dump, 1, FUNCTION_FLAGS),
  JS_FN("atob", AtoB, 1, FUNCTION_FLAGS),
  JS_FN("btoa", BtoA, 1, FUNCTION_FLAGS),
  JS_FS_END
};

const char* const WorkerGlobalScope::sEventStrings[STRING_COUNT] = {
  "onerror",
  "onclose"
};

class DedicatedWorkerGlobalScope : public WorkerGlobalScope
{
  static const DOMJSClass sClass;
  static const DOMIfaceAndProtoJSClass sProtoClass;
  static const JSPropertySpec sProperties[];
  static const JSFunctionSpec sFunctions[];

  enum
  {
    STRING_onmessage = 0,

    STRING_COUNT
  };

  static const char* const sEventStrings[STRING_COUNT];

public:
  static const JSClass*
  Class()
  {
    return sClass.ToJSClass();
  }

  static const JSClass*
  ProtoClass()
  {
    return sProtoClass.ToJSClass();
  }

  static const DOMClass*
  DOMClassStruct()
  {
    return &sClass.mClass;
  }

  static JSObject*
  InitClass(JSContext* aCx, JSObject* aObj, JSObject* aParentProto)
  {
    JS::Rooted<JSObject*> proto(aCx,
      JS_InitClass(aCx, aObj, aParentProto, ProtoClass(), Construct, 0,
                   sProperties, sFunctions, NULL, NULL));
    if (proto) {
      void* domClass = const_cast<DOMClass *>(DOMClassStruct());
      js::SetReservedSlot(proto, DOM_PROTO_INSTANCE_CLASS_SLOT,
                          JS::PrivateValue(domClass));
    }
    return proto;
  }

  static bool
  InitPrivate(JSContext* aCx, JSObject* aObj, WorkerPrivate* aWorkerPrivate)
  {
    JS_ASSERT(JS_GetClass(aObj) == Class());

    dom::AllocateProtoAndIfaceCache(aObj);

    nsRefPtr<DedicatedWorkerGlobalScope> scope =
      new DedicatedWorkerGlobalScope(aCx, aWorkerPrivate);

    js::SetReservedSlot(aObj, DOM_OBJECT_SLOT, PRIVATE_TO_JSVAL(scope));

    scope->SetIsDOMBinding();
    scope->SetWrapper(aObj);

    scope.forget();
    return true;
  }

protected:
  DedicatedWorkerGlobalScope(JSContext* aCx, WorkerPrivate* aWorker)
  : WorkerGlobalScope(aCx, aWorker)
  {
    MOZ_COUNT_CTOR(mozilla::dom::workers::DedicatedWorkerGlobalScope);
  }

  ~DedicatedWorkerGlobalScope()
  {
    MOZ_COUNT_DTOR(mozilla::dom::workers::DedicatedWorkerGlobalScope);
  }

private:
  using EventTarget::GetEventListener;
  using EventTarget::SetEventListener;

  static bool
  IsDedicatedWorkerGlobalScope(JS::Handle<JS::Value> v)
  {
    return v.isObject() && JS_GetClass(&v.toObject()) == Class();
  }

  static bool
  GetOnMessageImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
    const char* name = sEventStrings[STRING_onmessage];
    DedicatedWorkerGlobalScope* scope =
      GetInstancePrivate(aCx, &aArgs.thisv().toObject(), name);
    MOZ_ASSERT(scope);

    ErrorResult rv;

    nsRefPtr<EventHandlerNonNull> handler =
      scope->GetEventListener(NS_ConvertASCIItoUTF16(name + 2), rv);
    if (rv.Failed()) {
      JS_ReportError(aCx, "Failed to get event listener!");
      return false;
    }

    if (!handler) {
      aArgs.rval().setNull();
    } else {
      aArgs.rval().setObject(*handler->Callable());
    }
    return true;
  }

  static bool
  GetOnMessage(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
    JS::CallArgs args = JS::CallArgsFromVp(aArgc, aVp);
    return JS::CallNonGenericMethod<IsDedicatedWorkerGlobalScope, GetOnMessageImpl>(aCx, args);
  }

  static bool
  SetOnMessageImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
    const char* name = sEventStrings[STRING_onmessage];
    DedicatedWorkerGlobalScope* scope =
      GetInstancePrivate(aCx, &aArgs.thisv().toObject(), name);
    MOZ_ASSERT(scope);

    if (aArgs.length() == 0 || !aArgs[0].isObjectOrNull()) {
      JS_ReportError(aCx, "Not an event listener!");
      return false;
    }

    ErrorResult rv;

    JS::Rooted<JSObject*> listenerObj(aCx, aArgs[0].toObjectOrNull());
    nsRefPtr<EventHandlerNonNull> handler;
    if (listenerObj && JS_ObjectIsCallable(aCx, listenerObj)) {
      handler = new EventHandlerNonNull(listenerObj);
    } else {
      handler = nullptr;
    }
    scope->SetEventListener(NS_ConvertASCIItoUTF16(name + 2), handler, rv);

    if (rv.Failed()) {
      JS_ReportError(aCx, "Failed to set event listener!");
      return false;
    }

    aArgs.rval().setUndefined();
    return true;
  }

  static bool
  SetOnMessage(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
    JS::CallArgs args = JS::CallArgsFromVp(aArgc, aVp);
    return JS::CallNonGenericMethod<IsDedicatedWorkerGlobalScope, SetOnMessageImpl>(aCx, args);
  }

  static DedicatedWorkerGlobalScope*
  GetInstancePrivate(JSContext* aCx, JSObject* aObj, const char* aFunctionName)
  {
    const JSClass* classPtr = JS_GetClass(aObj);
    if (classPtr == Class()) {
      return UnwrapDOMObject<DedicatedWorkerGlobalScope>(aObj);
    }

    JS_ReportErrorNumber(aCx, js_GetErrorMessage, NULL,
                         JSMSG_INCOMPATIBLE_PROTO, Class()->name, aFunctionName,
                         classPtr->name);
    return NULL;
  }

  static bool
  Construct(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS_ReportErrorNumber(aCx, js_GetErrorMessage, NULL, JSMSG_WRONG_CONSTRUCTOR,
                         Class()->name);
    return false;
  }

  static bool
  Resolve(JSContext* aCx, JS::Handle<JSObject*> aObj, JS::Handle<jsid> aId,
          unsigned aFlags, JS::MutableHandle<JSObject*> aObjp)
  {
    bool resolved;
    if (!JS_ResolveStandardClass(aCx, aObj, aId, &resolved)) {
      return false;
    }

    aObjp.set(resolved ? aObj.get() : NULL);
    return true;
  }

  static void
  Finalize(JSFreeOp* aFop, JSObject* aObj)
  {
    JS_ASSERT(JS_GetClass(aObj) == Class());
    DedicatedWorkerGlobalScope* scope =
      UnwrapDOMObject<DedicatedWorkerGlobalScope>(aObj);
    if (scope) {
      DestroyProtoAndIfaceCache(aObj);
      scope->_finalize(aFop);
    }
  }

  static void
  Trace(JSTracer* aTrc, JSObject* aObj)
  {
    JS_ASSERT(JS_GetClass(aObj) == Class());
    DedicatedWorkerGlobalScope* scope =
      UnwrapDOMObject<DedicatedWorkerGlobalScope>(aObj);
    if (scope) {
      TraceProtoAndIfaceCache(aTrc, aObj);
      scope->_trace(aTrc);
    }
  }

  static bool
  PostMessage(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS::Rooted<JSObject*> obj(aCx, JS_THIS_OBJECT(aCx, aVp));
    if (!obj) {
      return false;
    }

    const char* name = sFunctions[0].name;
    DedicatedWorkerGlobalScope* scope = GetInstancePrivate(aCx, obj, name);
    if (!scope) {
      return false;
    }

    JS::Rooted<JS::Value> message(aCx);
    JS::Rooted<JS::Value> transferable(aCx, JSVAL_VOID);
    if (!JS_ConvertArguments(aCx, aArgc, JS_ARGV(aCx, aVp), "v/v",
                             message.address(), transferable.address())) {
      return false;
    }

    if (!scope->mWorker->PostMessageToParent(aCx, message, transferable)) {
      return false;
    }

    JS_RVAL(aCx, aVp).setUndefined();
    return true;
  }
};

const DOMJSClass DedicatedWorkerGlobalScope::sClass = {
  {
    // We don't have to worry about Xray expando slots here because we'll never
    // have an Xray wrapper to a worker global scope.
    "DedicatedWorkerGlobalScope",
    JSCLASS_DOM_GLOBAL | JSCLASS_IS_DOMJSCLASS | JSCLASS_IMPLEMENTS_BARRIERS |
    JSCLASS_GLOBAL_FLAGS_WITH_SLOTS(DOM_GLOBAL_SLOTS) | JSCLASS_NEW_RESOLVE,
    JS_PropertyStub, JS_DeletePropertyStub, JS_PropertyStub,
    JS_StrictPropertyStub, JS_EnumerateStub,
    reinterpret_cast<JSResolveOp>(Resolve), JS_ConvertStub, Finalize, nullptr,
    nullptr, nullptr, nullptr, Trace
  },
  {
    INTERFACE_CHAIN_1(prototypes::id::EventTarget_workers),
    false,
    &sWorkerNativePropertyHooks
  }
};

const DOMIfaceAndProtoJSClass DedicatedWorkerGlobalScope::sProtoClass = {
  {
    // XXXbz we use "DedicatedWorkerGlobalScope" here to match sClass
    // so that we can JS_InitClass this JSClass and then
    // call JS_NewObject with our sClass and have it find the right
    // prototype.
    "DedicatedWorkerGlobalScope",
    JSCLASS_IS_DOMIFACEANDPROTOJSCLASS | JSCLASS_HAS_RESERVED_SLOTS(2),
    JS_PropertyStub,       /* addProperty */
    JS_DeletePropertyStub, /* delProperty */
    JS_PropertyStub,       /* getProperty */
    JS_StrictPropertyStub, /* setProperty */
    JS_EnumerateStub,
    JS_ResolveStub,
    JS_ConvertStub,
    nullptr,               /* finalize */
    nullptr,               /* checkAccess */
    nullptr,               /* call */
    nullptr,               /* hasInstance */
    nullptr,               /* construct */
    nullptr,               /* trace */
    JSCLASS_NO_INTERNAL_MEMBERS
  },
  eInterfacePrototype,
  &sWorkerNativePropertyHooks,
  "[object DedicatedWorkerGlobalScope]",
  prototypes::id::_ID_Count,
  0
};

const JSPropertySpec DedicatedWorkerGlobalScope::sProperties[] = {
  JS_PSGS(sEventStrings[STRING_onmessage], GetOnMessage, SetOnMessage,
          JSPROP_ENUMERATE),
  JS_PS_END
};

const JSFunctionSpec DedicatedWorkerGlobalScope::sFunctions[] = {
  JS_FN("postMessage", PostMessage, 1, FUNCTION_FLAGS),
  JS_FS_END
};

const char* const DedicatedWorkerGlobalScope::sEventStrings[STRING_COUNT] = {
  "onmessage",
};

class SharedWorkerGlobalScope : public WorkerGlobalScope
{
  static DOMJSClass sClass;
  static DOMIfaceAndProtoJSClass sProtoClass;
  static const JSPropertySpec sProperties[];

  enum
  {
    STRING_onconnect = 0,

    STRING_COUNT
  };

  static const char* const sEventStrings[STRING_COUNT];

public:
  static const JSClass*
  Class()
  {
    return sClass.ToJSClass();
  }

  static const JSClass*
  ProtoClass()
  {
    return sProtoClass.ToJSClass();
  }

  static const DOMClass*
  DOMClassStruct()
  {
    return &sClass.mClass;
  }

  static JSObject*
  InitClass(JSContext* aCx, JSObject* aObj, JSObject* aParentProto)
  {
    JS::Rooted<JSObject*> proto(aCx,
      JS_InitClass(aCx, aObj, aParentProto, ProtoClass(), Construct, 0,
                   sProperties, nullptr, nullptr, nullptr));
    if (proto) {
      void* domClass = const_cast<DOMClass *>(DOMClassStruct());
      js::SetReservedSlot(proto, DOM_PROTO_INSTANCE_CLASS_SLOT,
                          JS::PrivateValue(domClass));
    }
    return proto;
  }

  static bool
  InitPrivate(JSContext* aCx, JSObject* aObj, WorkerPrivate* aWorkerPrivate)
  {
    MOZ_ASSERT(JS_GetClass(aObj) == Class());

    dom::AllocateProtoAndIfaceCache(aObj);

    nsRefPtr<SharedWorkerGlobalScope> scope =
      new SharedWorkerGlobalScope(aCx, aWorkerPrivate);

    js::SetReservedSlot(aObj, DOM_OBJECT_SLOT, PRIVATE_TO_JSVAL(scope));

    scope->SetIsDOMBinding();
    scope->SetWrapper(aObj);

    scope.forget();
    return true;
  }

protected:
  SharedWorkerGlobalScope(JSContext* aCx, WorkerPrivate* aWorker)
  : WorkerGlobalScope(aCx, aWorker)
  {
    MOZ_COUNT_CTOR(mozilla::dom::workers::SharedWorkerGlobalScope);
  }

  ~SharedWorkerGlobalScope()
  {
    MOZ_COUNT_DTOR(mozilla::dom::workers::SharedWorkerGlobalScope);
  }

private:
  using EventTarget::GetEventListener;
  using EventTarget::SetEventListener;

  static bool
  IsSharedWorkerGlobalScope(JS::Handle<JS::Value> aVal)
  {
    return aVal.isObject() && JS_GetClass(&aVal.toObject()) == Class();
  }

  static bool
  GetOnconnectImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
#if !defined(_MSC_VER) || _MSC_VER >= 1500
    auto name = sEventStrings[STRING_onconnect];
    auto scope = GetInstancePrivate(aCx, &aArgs.thisv().toObject(), name);
#else
    BOOST_AUTO(name, sEventStrings[STRING_onconnect]);
    BOOST_AUTO(scope, GetInstancePrivate(aCx, &aArgs.thisv().toObject(), name));
#endif

    MOZ_ASSERT(scope);

    ErrorResult rv;

    nsRefPtr<EventHandlerNonNull> handler =
      scope->GetEventListener(NS_ConvertASCIItoUTF16(name + 2), rv);
    if (rv.Failed()) {
      JS_ReportError(aCx, "Failed to get event listener!");
      return false;
    }

    if (!handler) {
      aArgs.rval().setNull();
    } else {
      aArgs.rval().setObject(*handler->Callable());
    }

    return true;
  }

  static bool
  GetOnconnect(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
#if !defined(_MSC_VER) || _MSC_VER >= 1500
    auto args = JS::CallArgsFromVp(aArgc, aVp);
#else
    BOOST_AUTO(args, JS::CallArgsFromVp(aArgc, aVp));
#endif
    return JS::CallNonGenericMethod<IsSharedWorkerGlobalScope,
                                    GetOnconnectImpl>(aCx, args);
  }

  static bool
  SetOnconnectImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
#if !defined(_MSC_VER) || _MSC_VER >= 1500
    auto name = sEventStrings[STRING_onconnect];
    auto scope = GetInstancePrivate(aCx, &aArgs.thisv().toObject(), name);
#else
    BOOST_AUTO(name, sEventStrings[STRING_onconnect]);
    BOOST_AUTO(scope, GetInstancePrivate(aCx, &aArgs.thisv().toObject(), name));
#endif
    MOZ_ASSERT(scope);

    if (aArgs.length() == 0 || !aArgs[0].isObject()) {
      JS_ReportError(aCx, "Not an event listener!");
      return false;
    }


    ErrorResult rv;

    JS::Rooted<JSObject*> listenerObj(aCx, aArgs[0].toObjectOrNull());
    nsRefPtr<EventHandlerNonNull> handler;
    if (listenerObj && JS_ObjectIsCallable(aCx, listenerObj)) {
      handler = new EventHandlerNonNull(listenerObj);
    } else {
      handler = nullptr;
    }
    scope->SetEventListener(NS_ConvertASCIItoUTF16(name + 2), handler, rv);

    if (rv.Failed()) {
      JS_ReportError(aCx, "Failed to set event listener!");
      return false;
    }

    aArgs.rval().setUndefined();
    return true;
  }

  static bool
  SetOnconnect(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
#if !defined(_MSC_VER) || _MSC_VER >= 1500
    auto args = JS::CallArgsFromVp(aArgc, aVp);
#else
    BOOST_AUTO(args, JS::CallArgsFromVp(aArgc, aVp));
#endif
    return JS::CallNonGenericMethod<IsSharedWorkerGlobalScope,
                                    SetOnconnectImpl>(aCx, args);
  }

  static bool
  GetNameImpl(JSContext* aCx, JS::CallArgs aArgs)
  {
#if !defined(_MSC_VER) || _MSC_VER >= 1500
    auto scope = GetInstancePrivate(aCx, &aArgs.thisv().toObject(), "name");
#else
    BOOST_AUTO(scope, GetInstancePrivate(aCx, &aArgs.thisv().toObject(), "name"));
#endif
    MOZ_ASSERT(scope);

#if !defined(_MSC_VER) || _MSC_VER >= 1500
    auto name = scope->mWorker->SharedWorkerName();
#else
    BOOST_AUTO(name, scope->mWorker->SharedWorkerName());
#endif
    MOZ_ASSERT(!name.IsVoid());

    JS::Rooted<JSString*> nameStr(aCx,
      JS_InternUCStringN(aCx, name.get(), name.Length()));
    if (!nameStr) {
      return false;
    }

    aArgs.rval().setString(nameStr);
    return true;
  }

  static bool
  GetName(JSContext* aCx, unsigned aArgc, JS::Value* aVp)
  {
#if !defined(_MSC_VER) || _MSC_VER >= 1500
    auto args = JS::CallArgsFromVp(aArgc, aVp);
#else
    BOOST_AUTO(args, JS::CallArgsFromVp(aArgc, aVp));
#endif
    return JS::CallNonGenericMethod<IsSharedWorkerGlobalScope,
                                    GetNameImpl>(aCx, args);
  }


  static SharedWorkerGlobalScope*
  GetInstancePrivate(JSContext* aCx, JSObject* aObj, const char* aFunctionName)
  {
    const JSClass* classPtr = JS_GetClass(aObj);
    if (classPtr == Class()) {
      return UnwrapDOMObject<SharedWorkerGlobalScope>(aObj);
    }

    JS_ReportErrorNumber(aCx, js_GetErrorMessage, nullptr,
                         JSMSG_INCOMPATIBLE_PROTO, Class()->name, aFunctionName,
                         classPtr->name);
    return nullptr;
  }

  static bool
  Construct(JSContext* aCx, unsigned aArgc, jsval* aVp)
  {
    JS_ReportErrorNumber(aCx, js_GetErrorMessage, nullptr,
                         JSMSG_WRONG_CONSTRUCTOR, Class()->name);
    return false;
  }

  static bool
  Resolve(JSContext* aCx, JS::Handle<JSObject*> aObj, JS::Handle<jsid> aId,
          unsigned aFlags, JS::MutableHandle<JSObject*> aObjp)
  {
    bool resolved;
    if (!JS_ResolveStandardClass(aCx, aObj, aId, &resolved)) {
      return false;
    }

    aObjp.set(resolved ? aObj.get() : nullptr);
    return true;
  }

  static void
  Finalize(JSFreeOp* aFop, JSObject* aObj)
  {
    MOZ_ASSERT(JS_GetClass(aObj) == Class());
    SharedWorkerGlobalScope* scope =
      UnwrapDOMObject<SharedWorkerGlobalScope>(aObj);
    if (scope) {
      DestroyProtoAndIfaceCache(aObj);
      scope->_finalize(aFop);
    }
  }

  static void
  Trace(JSTracer* aTrc, JSObject* aObj)
  {
    MOZ_ASSERT(JS_GetClass(aObj) == Class());
    SharedWorkerGlobalScope* scope =
      UnwrapDOMObject<SharedWorkerGlobalScope>(aObj);
    if (scope) {
      TraceProtoAndIfaceCache(aTrc, aObj);
      scope->_trace(aTrc);
    }
  }
};

DOMJSClass SharedWorkerGlobalScope::sClass = {
  {
    // We don't have to worry about Xray expando slots here because we'll never
    // have an Xray wrapper to a worker global scope.
    "SharedWorkerGlobalScope",
    JSCLASS_DOM_GLOBAL | JSCLASS_IS_DOMJSCLASS | JSCLASS_IMPLEMENTS_BARRIERS |
    JSCLASS_GLOBAL_FLAGS_WITH_SLOTS(DOM_GLOBAL_SLOTS) | JSCLASS_NEW_RESOLVE,
    JS_PropertyStub, JS_DeletePropertyStub, JS_PropertyStub,
    JS_StrictPropertyStub, JS_EnumerateStub,
    reinterpret_cast<JSResolveOp>(Resolve), JS_ConvertStub, Finalize, nullptr,
    nullptr, nullptr, nullptr, Trace
  },
  {
    INTERFACE_CHAIN_1(prototypes::id::EventTarget_workers),
    false,
    &sWorkerNativePropertyHooks
  }
};

DOMIfaceAndProtoJSClass SharedWorkerGlobalScope::sProtoClass = {
  {
    // XXXbz we use "SharedWorkerGlobalScope" here to match sClass
    // so that we can JS_InitClass this JSClass and then
    // call JS_NewObject with our sClass and have it find the right
    // prototype.
    "SharedWorkerGlobalScope",
    JSCLASS_IS_DOMIFACEANDPROTOJSCLASS | JSCLASS_HAS_RESERVED_SLOTS(2),
    JS_PropertyStub,       /* addProperty */
    JS_DeletePropertyStub, /* delProperty */
    JS_PropertyStub,       /* getProperty */
    JS_StrictPropertyStub, /* setProperty */
    JS_EnumerateStub,
    JS_ResolveStub,
    JS_ConvertStub,
    nullptr,               /* finalize */
    nullptr,               /* checkAccess */
    nullptr,               /* call */
    nullptr,               /* hasInstance */
    nullptr,               /* construct */
    nullptr,               /* trace */
    JSCLASS_NO_INTERNAL_MEMBERS
  },
  eInterfacePrototype,
  &sWorkerNativePropertyHooks,
  "[object SharedWorkerGlobalScope]",
  prototypes::id::_ID_Count,
  0
};

const JSPropertySpec SharedWorkerGlobalScope::sProperties[] = {
  JS_PSGS(sEventStrings[STRING_onconnect], GetOnconnect, SetOnconnect,
          JSPROP_ENUMERATE),
  JS_PSGS("name", GetName, GetterOnlyJSNative, JSPROP_ENUMERATE),
  JS_PS_END
};

const char* const SharedWorkerGlobalScope::sEventStrings[STRING_COUNT] = {
  "onconnect",
};

WorkerGlobalScope*
WorkerGlobalScope::GetInstancePrivate(JSContext* aCx, JSObject* aObj,
                                      const char* aFunctionName)
{
  const JSClass* classPtr = JS_GetClass(aObj);

  // We can only make [Dedicated|Shared]WorkerGlobalScope, not
  // WorkerGlobalScope, so this should never happen.
  MOZ_ASSERT(classPtr != Class());

  if (classPtr == DedicatedWorkerGlobalScope::Class()) {
    return UnwrapDOMObject<DedicatedWorkerGlobalScope>(aObj);
  }

  if (classPtr == SharedWorkerGlobalScope::Class()) {
    return UnwrapDOMObject<SharedWorkerGlobalScope>(aObj);
  }

  JS_ReportErrorNumber(aCx, js_GetErrorMessage, nullptr,
                       JSMSG_INCOMPATIBLE_PROTO, sClass.name, aFunctionName,
                       classPtr->name);
  return nullptr;
}

bool
WorkerGlobalScope::IsWorkerGlobalScope(JS::Handle<JS::Value> aVal)
{
  if (!aVal.isObject()) {
    return false;
  }

#if !defined(_MSC_VER) || _MSC_VER >= 1500
  auto classPtr = JS_GetClass(&aVal.toObject());
#else
  BOOST_AUTO(classPtr, JS_GetClass(&aVal.toObject()));
#endif

  return classPtr == DedicatedWorkerGlobalScope::Class() ||
         classPtr == SharedWorkerGlobalScope::Class();
}

} /* anonymous namespace */

BEGIN_WORKERS_NAMESPACE

JSObject*
CreateGlobalScope(JSContext* aCx)
{
  using namespace mozilla::dom;

  WorkerPrivate* worker = GetWorkerPrivateFromContext(aCx);
  MOZ_ASSERT(worker);

  const JSClass* classPtr = worker->IsSharedWorker() ?
                            SharedWorkerGlobalScope::Class() :
                            DedicatedWorkerGlobalScope::Class();

  JS::CompartmentOptions options;
  if (worker->IsChromeWorker()) {
    options.setVersion(JSVERSION_LATEST);
  }

  JS::Rooted<JSObject*> global(aCx,
    JS_NewGlobalObject(aCx, classPtr, GetWorkerPrincipal(),
                       JS::DontFireOnNewGlobalHook, options));
  if (!global) {
    return nullptr;
  }

  JSAutoCompartment ac(aCx, global);

  // Make the private slots now so that all our instance checks succeed.
  if (worker->IsSharedWorker()) {
    if (!SharedWorkerGlobalScope::InitPrivate(aCx, global, worker)) {
      return nullptr;
  }
  } else if (!DedicatedWorkerGlobalScope::InitPrivate(aCx, global, worker)) {
    return nullptr;
  }

  // Proto chain should be:
  //   global -> [Dedicated|Shared]WorkerGlobalScope
  //          -> WorkerGlobalScope
  //          -> EventTarget
  //          -> Object

  JS::Rooted<JSObject*> eventTargetProto(aCx,
    EventTargetBinding_workers::GetProtoObject(aCx, global));
  if (!eventTargetProto) {
    return nullptr;
  }

  JS::Rooted<JSObject*> scopeProto(aCx,
    WorkerGlobalScope::InitClass(aCx, global, eventTargetProto));
  if (!scopeProto) {
    return nullptr;
  }

  JS::Rooted<JSObject*> finalScopeProto(aCx,
    worker->IsSharedWorker() ?
    SharedWorkerGlobalScope::InitClass(aCx, global, scopeProto) :
    DedicatedWorkerGlobalScope::InitClass(aCx, global, scopeProto));
  if (!finalScopeProto) {
    return nullptr;
  }

  if (!JS_SetPrototype(aCx, global, finalScopeProto)) {
    return nullptr;
  }

  JS::Rooted<JSObject*> workerProto(aCx,
    worker::InitClass(aCx, global, eventTargetProto, false));
  if (!workerProto) {
    return nullptr;
  }

  if (worker->IsChromeWorker()) {
    if (!chromeworker::InitClass(aCx, global, workerProto, false) ||
        !DefineChromeWorkerFunctions(aCx, global) ||
        !DefineOSFileConstants(aCx, global)) {
      return nullptr;
    }
  }

  // Init other classes we care about.
  if (!events::InitClasses(aCx, global, false) ||
      !file::InitClasses(aCx, global)) {
    return nullptr;
  }

  // Init other paris-bindings.
  if (!DOMExceptionBinding::GetConstructorObject(aCx, global) ||
      !EventBinding::GetConstructorObject(aCx, global) ||
      !FileReaderSyncBinding_workers::GetConstructorObject(aCx, global) ||
      !ImageDataBinding::GetConstructorObject(aCx, global) ||
      !TextDecoderBinding::GetConstructorObject(aCx, global) ||
      !TextEncoderBinding::GetConstructorObject(aCx, global) ||
      !XMLHttpRequestBinding_workers::GetConstructorObject(aCx, global) ||
      !XMLHttpRequestUploadBinding_workers::GetConstructorObject(aCx, global) ||
      !URLBinding_workers::GetConstructorObject(aCx, global) ||
      !WorkerLocationBinding_workers::GetConstructorObject(aCx, global) ||
      !WorkerNavigatorBinding_workers::GetConstructorObject(aCx, global)) {
    return nullptr;
  }

  if (!JS_DefineProfilingFunctions(aCx, global)) {
    return nullptr;
  }

  JS_FireOnNewGlobalObject(aCx, global);

  return global;
}

END_WORKERS_NAMESPACE
