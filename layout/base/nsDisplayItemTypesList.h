// IWYU pragma: private, include "nsDisplayList.h"
DECLARE_DISPLAY_ITEM_TYPE(ALT_FEEDBACK)
DECLARE_DISPLAY_ITEM_TYPE(BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(THEMED_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE_FLAGS(BACKGROUND_COLOR,TYPE_RENDERS_NO_IMAGES)
DECLARE_DISPLAY_ITEM_TYPE(BLEND_CONTAINER)
DECLARE_DISPLAY_ITEM_TYPE(BORDER)
DECLARE_DISPLAY_ITEM_TYPE(BOX_SHADOW_OUTER)
DECLARE_DISPLAY_ITEM_TYPE(BOX_SHADOW_INNER)
DECLARE_DISPLAY_ITEM_TYPE(BULLET)
DECLARE_DISPLAY_ITEM_TYPE(BUTTON_BORDER_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(BUTTON_BOX_SHADOW_OUTER)
DECLARE_DISPLAY_ITEM_TYPE(BUTTON_FOREGROUND)
DECLARE_DISPLAY_ITEM_TYPE(CANVAS)
DECLARE_DISPLAY_ITEM_TYPE(CANVAS_BACKGROUND_COLOR)
DECLARE_DISPLAY_ITEM_TYPE(CANVAS_THEMED_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(CANVAS_BACKGROUND_IMAGE)
DECLARE_DISPLAY_ITEM_TYPE(CANVAS_FOCUS)
DECLARE_DISPLAY_ITEM_TYPE(CARET)
DECLARE_DISPLAY_ITEM_TYPE(CHECKED_CHECKBOX)
DECLARE_DISPLAY_ITEM_TYPE(CHECKED_RADIOBUTTON)
DECLARE_DISPLAY_ITEM_TYPE(COLUMN_RULE)
DECLARE_DISPLAY_ITEM_TYPE(COMBOBOX_FOCUS)
DECLARE_DISPLAY_ITEM_TYPE(EVENT_RECEIVER)
DECLARE_DISPLAY_ITEM_TYPE(FIELDSET_BORDER_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(FIXED_POSITION)
DECLARE_DISPLAY_ITEM_TYPE(FORCEPAINTONSCROLL)
DECLARE_DISPLAY_ITEM_TYPE(FRAMESET_BORDER)
DECLARE_DISPLAY_ITEM_TYPE(FRAMESET_BLANK)
DECLARE_DISPLAY_ITEM_TYPE(HEADER_FOOTER)
DECLARE_DISPLAY_ITEM_TYPE(IMAGE)
DECLARE_DISPLAY_ITEM_TYPE(LIST_FOCUS)
DECLARE_DISPLAY_ITEM_TYPE(MIX_BLEND_MODE)
DECLARE_DISPLAY_ITEM_TYPE(OPACITY)
DECLARE_DISPLAY_ITEM_TYPE(OPTION_EVENT_GRABBER)
DECLARE_DISPLAY_ITEM_TYPE(OUTLINE)
DECLARE_DISPLAY_ITEM_TYPE(OWN_LAYER)
DECLARE_DISPLAY_ITEM_TYPE(PAGE_CONTENT)
DECLARE_DISPLAY_ITEM_TYPE(PAGE_SEQUENCE)
DECLARE_DISPLAY_ITEM_TYPE(PLUGIN)
DECLARE_DISPLAY_ITEM_TYPE(PLUGIN_READBACK)
DECLARE_DISPLAY_ITEM_TYPE(PLUGIN_VIDEO)
DECLARE_DISPLAY_ITEM_TYPE(PRINT_PLUGIN)
DECLARE_DISPLAY_ITEM_TYPE(REMOTE)
DECLARE_DISPLAY_ITEM_TYPE(REMOTE_SHADOW)
DECLARE_DISPLAY_ITEM_TYPE(SCROLL_LAYER)
DECLARE_DISPLAY_ITEM_TYPE(SCROLL_INFO_LAYER)
DECLARE_DISPLAY_ITEM_TYPE(SELECTION_OVERLAY)
DECLARE_DISPLAY_ITEM_TYPE(SOLID_COLOR)
DECLARE_DISPLAY_ITEM_TYPE(SVG_EFFECTS)
DECLARE_DISPLAY_ITEM_TYPE(SVG_GLYPHS)
DECLARE_DISPLAY_ITEM_TYPE(SVG_OUTER_SVG)
DECLARE_DISPLAY_ITEM_TYPE(SVG_PATH_GEOMETRY)
DECLARE_DISPLAY_ITEM_TYPE(SVG_TEXT)
DECLARE_DISPLAY_ITEM_TYPE(TABLE_CELL_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(TABLE_CELL_SELECTION)
DECLARE_DISPLAY_ITEM_TYPE(TABLE_ROW_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(TABLE_ROW_GROUP_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(TABLE_BORDER_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(TEXT)
DECLARE_DISPLAY_ITEM_TYPE(TEXT_DECORATION)
DECLARE_DISPLAY_ITEM_TYPE(TEXT_OVERFLOW)
DECLARE_DISPLAY_ITEM_TYPE(TEXT_SHADOW)
DECLARE_DISPLAY_ITEM_TYPE(TRANSFORM)
DECLARE_DISPLAY_ITEM_TYPE(VIDEO)
DECLARE_DISPLAY_ITEM_TYPE(WRAP_LIST)
DECLARE_DISPLAY_ITEM_TYPE(ZOOM)
DECLARE_DISPLAY_ITEM_TYPE(EXCLUDE_GLASS_FRAME)

#if defined(MOZ_REFLOW_PERF_DSP) && defined(MOZ_REFLOW_PERF)
DECLARE_DISPLAY_ITEM_TYPE(REFLOW_COUNT)
#endif

#ifdef MOZ_XUL
DECLARE_DISPLAY_ITEM_TYPE(XUL_EVENT_REDIRECTOR)
DECLARE_DISPLAY_ITEM_TYPE(XUL_GROUP_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(XUL_IMAGE)
DECLARE_DISPLAY_ITEM_TYPE(XUL_TEXT_BOX)
DECLARE_DISPLAY_ITEM_TYPE(XUL_TREE_BODY)
DECLARE_DISPLAY_ITEM_TYPE(XUL_TREE_COL_SPLITTER_TARGET)
#ifdef DEBUG_LAYOUT
DECLARE_DISPLAY_ITEM_TYPE(XUL_DEBUG)
#endif
#endif

DECLARE_DISPLAY_ITEM_TYPE(MATHML_BAR)
DECLARE_DISPLAY_ITEM_TYPE(MATHML_CHAR_BACKGROUND)
DECLARE_DISPLAY_ITEM_TYPE(MATHML_CHAR_FOREGROUND)
DECLARE_DISPLAY_ITEM_TYPE(MATHML_ERROR)
DECLARE_DISPLAY_ITEM_TYPE(MATHML_MENCLOSE_NOTATION)
DECLARE_DISPLAY_ITEM_TYPE(MATHML_SELECTION_RECT)
DECLARE_DISPLAY_ITEM_TYPE(MATHML_SLASH)
#ifdef DEBUG
DECLARE_DISPLAY_ITEM_TYPE(MATHML_BOUNDING_METRICS)
DECLARE_DISPLAY_ITEM_TYPE(MATHML_CHAR_DEBUG)

DECLARE_DISPLAY_ITEM_TYPE(DEBUG_BORDER)
DECLARE_DISPLAY_ITEM_TYPE(DEBUG_IMAGE_MAP)
DECLARE_DISPLAY_ITEM_TYPE(DEBUG_PLACEHOLDER)
DECLARE_DISPLAY_ITEM_TYPE(EVENT_TARGET_BORDER)
#endif