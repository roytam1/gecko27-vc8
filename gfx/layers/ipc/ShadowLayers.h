/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: sw=2 ts=8 et :
 */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_layers_ShadowLayers_h
#define mozilla_layers_ShadowLayers_h 1

#include <stddef.h>                     // for size_t
#include "mozilla/StandardInteger.h"    // for uint64_t
#include "gfxTypes.h"
#include "gfxPoint.h"                   // for gfxIntSize
#include "mozilla/Attributes.h"         // for MOZ_OVERRIDE
#include "mozilla/WidgetUtils.h"        // for ScreenRotation
#include "mozilla/dom/ScreenOrientation.h"  // for ScreenOrientation
#include "mozilla/ipc/SharedMemory.h"   // for SharedMemory, etc
#include "mozilla/layers/CompositableForwarder.h"
#include "mozilla/layers/CompositorTypes.h"  // for OpenMode, etc
#include "nsCOMPtr.h"                   // for already_AddRefed
#include "nsRegion.h"                   // for nsIntRegion
#include "nsTArrayForwardDeclare.h"     // for InfallibleTArray
 
struct nsIntPoint;
struct nsIntRect;
class gfxASurface;

namespace mozilla {
namespace layers {

class BasicTiledLayerBuffer;
class CanvasClient;
class CanvasLayerComposite;
class CanvasSurface;
class ColorLayerComposite;
class CompositableChild;
class ContainerLayerComposite;
class ContentClient;
class ContentClientRemote;
class EditReply;
class ImageClient;
class ImageLayerComposite;
class Layer;
class OptionalThebesBuffer;
class PLayerChild;
class PLayerTransactionChild;
class PLayerTransactionParent;
class RefLayerComposite;
class ShadowableLayer;
class Shmem;
class ShmemTextureClient;
class SurfaceDescriptor;
class TextureClient;
class ThebesLayerComposite;
class ThebesBuffer;
class ThebesBufferData;
class TiledLayerComposer;
class Transaction;


/**
 * We want to share layer trees across thread contexts and address
 * spaces for several reasons; chief among them
 *
 *  - a parent process can paint a child process's layer tree while
 *    the child process is blocked, say on content script.  This is
 *    important on mobile devices where UI responsiveness is key.
 *
 *  - a dedicated "compositor" process can asynchronously (wrt the
 *    browser process) composite and animate layer trees, allowing a
 *    form of pipeline parallelism between compositor/browser/content
 *
 *  - a dedicated "compositor" process can take all responsibility for
 *    accessing the GPU, which is desirable on systems with
 *    buggy/leaky drivers because the compositor process can die while
 *    browser and content live on (and failover mechanisms can be
 *    installed to quickly bring up a replacement compositor)
 *
 * The Layers model has a crisply defined API, which makes it easy to
 * safely "share" layer trees.  The ShadowLayers API extends Layers to
 * allow a remote, parent process to access a child process's layer
 * tree.
 *
 * ShadowLayerForwarder publishes a child context's layer tree to a
 * parent context.  This comprises recording layer-tree modifications
 * into atomic transactions and pushing them over IPC.
 *
 * LayerManagerComposite grafts layer subtrees published by child-context
 * ShadowLayerForwarder(s) into a parent-context layer tree.
 *
 * (Advanced note: because our process tree may have a height >2, a
 * non-leaf subprocess may both receive updates from child processes
 * and publish them to parent processes.  Put another way,
 * LayerManagers may be both LayerManagerComposites and
 * ShadowLayerForwarders.)
 *
 * There are only shadow types for layers that have different shadow
 * vs. not-shadow behavior.  ColorLayers and ContainerLayers behave
 * the same way in both regimes (so far).
 *
 *
 * The mecanism to shadow the layer tree on the compositor through IPC works as
 * follows:
 * The layer tree is managed on the content thread, and shadowed in the compositor
 * thread. The shadow layer tree is only kept in sync with whatever happens in
 * the content thread. To do this we use IPDL protocols. IPDL is a domain
 * specific language that describes how two processes or thread should
 * communicate. C++ code is generated from .ipdl files to implement the message
 * passing, synchronization and serialization logic. To use the generated code
 * we implement classes that inherit the generated IPDL actor. the ipdl actors
 * of a protocol PX are PXChild or PXParent (the generated class), and we
 * conventionally implement XChild and XParent. The Parent side of the protocol
 * is the one that lives on the compositor thread. Think of IPDL actors as
 * endpoints of communication. they are useful to send messages and also to
 * dispatch the message to the right actor on the other side. One nice property
 * of an IPDL actor is that when an actor, say PXChild is sent in a message, the
 * PXParent comes out in the other side. we use this property a lot to dispatch
 * messages to the right layers and compositable, each of which have their own
 * ipdl actor on both side.
 *
 * Most of the synchronization logic happens in layer transactions and
 * compositable transactions.
 * A transaction is a set of changes to the layers and/or the compositables
 * that are sent and applied together to the compositor thread to keep the
 * LayerComposite in a coherent state.
 * Layer transactions maintain the shape of the shadow layer tree, and
 * synchronize the texture data held by compositables. Layer transactions
 * are always between the content thread and the compositor thread.
 * Compositable transactions are subset of a layer transaction with which only
 * compositables and textures can be manipulated, and does not always originate
 * from the content thread. (See CompositableForwarder.h and ImageBridgeChild.h)
 */

class ShadowLayerForwarder : public CompositableForwarder
{
  friend class AutoOpenSurface;
  friend class DeprecatedTextureClientShmem;
  friend class ContentClientIncremental;

public:
  virtual ~ShadowLayerForwarder();

  /**
   * Setup the IPDL actor for aCompositable to be part of layers
   * transactions.
   */
  void Connect(CompositableClient* aCompositable);

  virtual void CreatedSingleBuffer(CompositableClient* aCompositable,
                                   const SurfaceDescriptor& aDescriptor,
                                   const TextureInfo& aTextureInfo,
                                   const SurfaceDescriptor* aDescriptorOnWhite = nullptr) MOZ_OVERRIDE;
  virtual void CreatedIncrementalBuffer(CompositableClient* aCompositable,
                                        const TextureInfo& aTextureInfo,
                                        const nsIntRect& aBufferRect) MOZ_OVERRIDE;
  virtual void CreatedDoubleBuffer(CompositableClient* aCompositable,
                                   const SurfaceDescriptor& aFrontDescriptor,
                                   const SurfaceDescriptor& aBackDescriptor,
                                   const TextureInfo& aTextureInfo,
                                   const SurfaceDescriptor* aFrontDescriptorOnWhite = nullptr,
                                   const SurfaceDescriptor* aBackDescriptorOnWhite = nullptr) MOZ_OVERRIDE;
  virtual void DestroyThebesBuffer(CompositableClient* aCompositable) MOZ_OVERRIDE;

  /**
   * Adds an edit in the layers transaction in order to attach
   * the corresponding compositable and layer on the compositor side.
   * Connect must have been called on aCompositable beforehand.
   */
  void Attach(CompositableClient* aCompositable,
              ShadowableLayer* aLayer);

  /**
   * Adds an edit in the transaction in order to attach a Compositable that
   * is not managed by this ShadowLayerForwarder (for example, by ImageBridge
   * in the case of async-video).
   * Since the compositable is not managed by this forwarder, we can't use
   * the compositable or it's IPDL actor here, so we use an ID instead, that
   * is matched on the compositor side.
   */
  void AttachAsyncCompositable(uint64_t aCompositableID,
                               ShadowableLayer* aLayer);

  /**
   * Begin recording a transaction to be forwarded atomically to a
   * LayerManagerComposite.
   */
  void BeginTransaction(const nsIntRect& aTargetBounds,
                        ScreenRotation aRotation,
                        const nsIntRect& aClientBounds,
                        mozilla::dom::ScreenOrientation aOrientation);

  /**
   * The following methods may only be called after BeginTransaction()
   * but before EndTransaction().  They mirror the LayerManager
   * interface in Layers.h.
   */

  /**
   * Notify the shadow manager that a new, "real" layer has been
   * created, and a corresponding shadow layer should be created in
   * the compositing process.
   */
  void CreatedThebesLayer(ShadowableLayer* aThebes);
  void CreatedContainerLayer(ShadowableLayer* aContainer);
  void CreatedImageLayer(ShadowableLayer* aImage);
  void CreatedColorLayer(ShadowableLayer* aColor);
  void CreatedCanvasLayer(ShadowableLayer* aCanvas);
  void CreatedRefLayer(ShadowableLayer* aRef);

  /**
   * The specified layer is destroying its buffers.
   * |aBackBufferToDestroy| is deallocated when this transaction is
   * posted to the parent.  During the parent-side transaction, the
   * shadow is told to destroy its front buffer.  This can happen when
   * a new front/back buffer pair have been created because of a layer
   * resize, e.g.
   */
  virtual void DestroyedThebesBuffer(const SurfaceDescriptor& aBackBufferToDestroy) MOZ_OVERRIDE;

  /**
   * At least one attribute of |aMutant| has changed, and |aMutant|
   * needs to sync to its shadow layer.  This initial implementation
   * forwards all attributes when any is mutated.
   */
  void Mutated(ShadowableLayer* aMutant);

  void SetRoot(ShadowableLayer* aRoot);
  /**
   * Insert |aChild| after |aAfter| in |aContainer|.  |aAfter| can be
   * nullptr to indicated that |aChild| should be appended to the end of
   * |aContainer|'s child list.
   */
  void InsertAfter(ShadowableLayer* aContainer,
                   ShadowableLayer* aChild,
                   ShadowableLayer* aAfter = nullptr);
  void RemoveChild(ShadowableLayer* aContainer,
                   ShadowableLayer* aChild);
  void RepositionChild(ShadowableLayer* aContainer,
                       ShadowableLayer* aChild,
                       ShadowableLayer* aAfter = nullptr);

  /**
   * Set aMaskLayer as the mask on aLayer.
   * Note that only image layers are properly supported
   * LayerTransactionParent::UpdateMask and accompanying ipdl
   * will need changing to update properties for other kinds
   * of mask layer.
   */
  void SetMask(ShadowableLayer* aLayer,
               ShadowableLayer* aMaskLayer);

  /**
   * Notify the compositor that a tiled layer buffer has changed
   * that needs to be synced to the shadow retained copy. The tiled
   * layer buffer will operate directly on the shadow retained buffer
   * and is free to choose it's own internal representation (double buffering,
   * copy on write, tiling).
   */
  virtual void PaintedTiledLayerBuffer(CompositableClient* aCompositable,
                                       const SurfaceDescriptorTiles& aTileLayerDescriptor) MOZ_OVERRIDE;

  /**
   * Notify the compositor that a compositable will be updated asynchronously
   * through ImageBridge, using an ID to connect the protocols on the
   * compositor side.
   */
  void AttachAsyncCompositable(PLayerTransactionChild* aLayer, uint64_t aID);

  /**
   * Communicate to the compositor that the texture identified by aLayer
   * and aIdentifier has been updated to aImage.
   */
  virtual void UpdateTexture(CompositableClient* aCompositable,
                             TextureIdentifier aTextureId,
                             SurfaceDescriptor* aDescriptor) MOZ_OVERRIDE;

  /**
   * Same as above, but performs an asynchronous layer transaction
   */
  virtual void UpdateTextureNoSwap(CompositableClient* aCompositable,
                                   TextureIdentifier aTextureId,
                                   SurfaceDescriptor* aDescriptor) MOZ_OVERRIDE;

  /**
   * Communicate to the compositor that aRegion in the texture identified by aLayer
   * and aIdentifier has been updated to aThebesBuffer.
   */
  virtual void UpdateTextureRegion(CompositableClient* aCompositable,
                                   const ThebesBufferData& aThebesBufferData,
                                   const nsIntRegion& aUpdatedRegion) MOZ_OVERRIDE;

  virtual void UpdateTextureIncremental(CompositableClient* aCompositable,
                                        TextureIdentifier aTextureId,
                                        SurfaceDescriptor& aDescriptor,
                                        const nsIntRegion& aUpdatedRegion,
                                        const nsIntRect& aBufferRect,
                                        const nsIntPoint& aBufferRotation) MOZ_OVERRIDE;

  /**
   * Communicate the picture rect of an image to the compositor
   */
  void UpdatePictureRect(CompositableClient* aCompositable,
                         const nsIntRect& aRect);

  /**
   * See CompositableForwarder::AddTexture
   */
  virtual bool AddTexture(CompositableClient* aCompositable,
                          TextureClient* aClient) MOZ_OVERRIDE;

  /**
   * See CompositableForwarder::RemoveTexture
   */
  virtual void RemoveTexture(CompositableClient* aCompositable,
                             uint64_t aTextureID,
                             TextureFlags aFlags) MOZ_OVERRIDE;

  /**
   * See CompositableForwarder::UpdatedTexture
   */
  virtual void UpdatedTexture(CompositableClient* aCompositable,
                              TextureClient* aTexture,
                              nsIntRegion* aRegion) MOZ_OVERRIDE;

  /**
   * See CompositableForwarder::UseTexture
   */
  virtual void UseTexture(CompositableClient* aCompositable,
                          TextureClient* aClient) MOZ_OVERRIDE;

  /**
   * End the current transaction and forward it to LayerManagerComposite.
   * |aReplies| are directions from the LayerManagerComposite to the
   * caller of EndTransaction().
   */
  bool EndTransaction(InfallibleTArray<EditReply>* aReplies, bool* aSent);

  /**
   * Set an actor through which layer updates will be pushed.
   */
  void SetShadowManager(PLayerTransactionChild* aShadowManager)
  {
    mShadowManager = aShadowManager;
  }

  /**
   * True if this is forwarding to a LayerManagerComposite.
   */
  bool HasShadowManager() const { return !!mShadowManager; }
  PLayerTransactionChild* GetShadowManager() const { return mShadowManager; }

  virtual void WindowOverlayChanged() { mWindowOverlayChanged = true; }

  /**
   * The following Alloc/Open/Destroy interfaces abstract over the
   * details of working with surfaces that are shared across
   * processes.  They provide the glue between C++ Layers and the
   * LayerComposite IPC system.
   *
   * The basic lifecycle is
   *
   *  - a Layer needs a buffer.  Its ShadowableLayer subclass calls
   *    AllocBuffer(), then calls one of the Created*Buffer() methods
   *    above to transfer the (temporary) front buffer to its
   *    LayerComposite in the other process.  The Layer needs a
   *    gfxASurface to paint, so the ShadowableLayer uses
   *    OpenDescriptor(backBuffer) to get that surface, and hands it
   *    out to the Layer.
   *
   * - a Layer has painted new pixels.  Its ShadowableLayer calls one
   *   of the Painted*Buffer() methods above with the back buffer
   *   descriptor.  This notification is forwarded to the LayerComposite,
   *   which uses OpenDescriptor() to access the newly-painted pixels.
   *   The LayerComposite then updates its front buffer in a Layer- and
   *   platform-dependent way, and sends a surface descriptor back to
   *   the ShadowableLayer that becomes its new back back buffer.
   *
   * - a Layer wants to destroy its buffers.  Its ShadowableLayer
   *   calls Destroyed*Buffer(), which gives up control of the back
   *   buffer descriptor.  The actual back buffer surface is then
   *   destroyed using DestroySharedSurface() just before notifying
   *   the parent process.  When the parent process is notified, the
   *   LayerComposite also calls DestroySharedSurface() on its front
   *   buffer, and the double-buffer pair is gone.
   */

  // ISurfaceAllocator
  virtual bool AllocUnsafeShmem(size_t aSize,
                                ipc::SharedMemory::SharedMemoryType aType,
                                ipc::Shmem* aShmem) MOZ_OVERRIDE;
  virtual bool AllocShmem(size_t aSize,
                          ipc::SharedMemory::SharedMemoryType aType,
                          ipc::Shmem* aShmem) MOZ_OVERRIDE;
  virtual void DeallocShmem(ipc::Shmem& aShmem) MOZ_OVERRIDE;

  /**
   * Construct a shadow of |aLayer| on the "other side", at the
   * LayerManagerComposite.
   */
  PLayerChild* ConstructShadowFor(ShadowableLayer* aLayer);

  /**
   * Flag the next paint as the first for a document.
   */
  void SetIsFirstPaint() { mIsFirstPaint = true; }

  static void PlatformSyncBeforeUpdate();

  static already_AddRefed<gfxASurface>
  OpenDescriptor(OpenMode aMode, const SurfaceDescriptor& aSurface);

protected:
  ShadowLayerForwarder();

#ifdef DEBUG
  void CheckSurfaceDescriptor(const SurfaceDescriptor* aDescriptor) const;
#else
  void CheckSurfaceDescriptor(const SurfaceDescriptor* aDescriptor) const {}
#endif

  PLayerTransactionChild* mShadowManager;

#ifdef MOZ_HAVE_SURFACEDESCRIPTORGRALLOC
  // from ISurfaceAllocator
  virtual PGrallocBufferChild* AllocGrallocBuffer(const gfxIntSize& aSize,
                                                  uint32_t aFormat,
                                                  uint32_t aUsage,
                                                  MaybeMagicGrallocBufferHandle* aHandle) MOZ_OVERRIDE;
#endif

private:
  /**
   * Try to query the content type efficiently, but at worst map the
   * surface and return it in *aSurface.
   */
  static gfxContentType
  GetDescriptorSurfaceContentType(const SurfaceDescriptor& aDescriptor,
                                  OpenMode aMode,
                                  gfxASurface** aSurface);
  /**
   * It can be expensive to open a descriptor just to query its
   * content type.  If the platform impl can do this cheaply, it will
   * set *aContent and return true.
   */
  static bool
  PlatformGetDescriptorSurfaceContentType(const SurfaceDescriptor& aDescriptor,
                                          OpenMode aMode,
                                          gfxContentType* aContent,
                                          gfxASurface** aSurface);
  // (Same as above, but for surface size.)
  static gfxIntSize
  GetDescriptorSurfaceSize(const SurfaceDescriptor& aDescriptor,
                           OpenMode aMode,
                           gfxASurface** aSurface);
  static bool
  PlatformGetDescriptorSurfaceSize(const SurfaceDescriptor& aDescriptor,
                                   OpenMode aMode,
                                   gfxIntSize* aSize,
                                   gfxASurface** aSurface);
  // And again, for the image format.
  // This function will return gfxImageFormatUnknown only if |aDescriptor|
  // describes a non-ImageSurface.
  static gfxImageFormat
  GetDescriptorSurfaceImageFormat(const SurfaceDescriptor& aDescriptor,
                                  OpenMode aMode,
                                  gfxASurface** aSurface);
  static bool
  PlatformGetDescriptorSurfaceImageFormat(const SurfaceDescriptor& aDescriptor,
                                          OpenMode aMode,
                                          gfxImageFormat* aContent,
                                          gfxASurface** aSurface);

  static already_AddRefed<gfxASurface>
  PlatformOpenDescriptor(OpenMode aMode, const SurfaceDescriptor& aDescriptor);

  /**
   * Make this descriptor unusable for gfxASurface clients. A
   * private interface with AutoOpenSurface.
   */
  static void
  CloseDescriptor(const SurfaceDescriptor& aDescriptor);

  static bool
  PlatformCloseDescriptor(const SurfaceDescriptor& aDescriptor);

  bool PlatformDestroySharedSurface(SurfaceDescriptor* aSurface);

  Transaction* mTxn;
  DiagnosticTypes mDiagnosticTypes;
  bool mIsFirstPaint;
  bool mWindowOverlayChanged;
};

class CompositableClient;

/**
 * A ShadowableLayer is a Layer can be shared with a parent context
 * through a ShadowLayerForwarder.  A ShadowableLayer maps to a
 * Shadow*Layer in a parent context.
 *
 * Note that ShadowLayers can themselves be ShadowableLayers.
 */
class ShadowableLayer
{
public:
  virtual ~ShadowableLayer() {}

  virtual Layer* AsLayer() = 0;

  /**
   * True if this layer has a shadow in a parent process.
   */
  bool HasShadow() { return !!mShadow; }

  /**
   * Return the IPC handle to a Shadow*Layer referring to this if one
   * exists, nullptr if not.
   */
  PLayerChild* GetShadow() { return mShadow; }

  virtual CompositableClient* GetCompositableClient() { return nullptr; }
protected:
  ShadowableLayer() : mShadow(nullptr) {}

  PLayerChild* mShadow;
};

} // namespace layers
} // namespace mozilla

#endif // ifndef mozilla_layers_ShadowLayers_h
