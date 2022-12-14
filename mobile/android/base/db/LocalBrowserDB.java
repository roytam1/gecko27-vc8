/* -*- Mode: Java; c-basic-offset: 4; tab-width: 20; indent-tabs-mode: nil; -*-
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

package org.mozilla.gecko.db;

import org.mozilla.gecko.AboutPages;
import org.mozilla.gecko.db.BrowserContract.Bookmarks;
import org.mozilla.gecko.db.BrowserContract.Combined;
import org.mozilla.gecko.db.BrowserContract.ExpirePriority;
import org.mozilla.gecko.db.BrowserContract.FaviconColumns;
import org.mozilla.gecko.db.BrowserContract.Favicons;
import org.mozilla.gecko.db.BrowserContract.History;
import org.mozilla.gecko.db.BrowserContract.SyncColumns;
import org.mozilla.gecko.db.BrowserContract.Thumbnails;
import org.mozilla.gecko.db.BrowserContract.URLColumns;
import org.mozilla.gecko.gfx.BitmapUtils;

import android.content.ContentProviderOperation;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.database.ContentObserver;
import android.database.Cursor;
import android.database.CursorWrapper;
import android.database.DatabaseUtils;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.provider.Browser;
import android.text.TextUtils;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

public class LocalBrowserDB implements BrowserDB.BrowserDBIface {
    // Calculate these once, at initialization. isLoggable is too expensive to
    // have in-line in each log call.
    private static final String LOGTAG = "GeckoLocalBrowserDB";
    private static boolean logDebug = Log.isLoggable(LOGTAG, Log.DEBUG);
    protected static void debug(String message) {
        if (logDebug) {
            Log.d(LOGTAG, message);
        }
    }

    private final String mProfile;

    // Map of folder GUIDs to IDs. Used for caching.
    private HashMap<String, Long> mFolderIdMap;

    // Use wrapped Boolean so that we can have a null state
    private Boolean mDesktopBookmarksExist;

    private final Uri mBookmarksUriWithProfile;
    private final Uri mParentsUriWithProfile;
    private final Uri mHistoryUriWithProfile;
    private final Uri mHistoryExpireUriWithProfile;
    private final Uri mCombinedUriWithProfile;
    private final Uri mDeletedHistoryUriWithProfile;
    private final Uri mUpdateHistoryUriWithProfile;
    private final Uri mFaviconsUriWithProfile;
    private final Uri mThumbnailsUriWithProfile;

    private static final String[] DEFAULT_BOOKMARK_COLUMNS =
            new String[] { Bookmarks._ID,
                           Bookmarks.GUID,
                           Bookmarks.URL,
                           Bookmarks.TITLE,
                           Bookmarks.TYPE,
                           Bookmarks.PARENT };

    public LocalBrowserDB(String profile) {
        mProfile = profile;
        mFolderIdMap = new HashMap<String, Long>();
        mDesktopBookmarksExist = null;

        mBookmarksUriWithProfile = appendProfile(Bookmarks.CONTENT_URI);
        mParentsUriWithProfile = appendProfile(Bookmarks.PARENTS_CONTENT_URI);
        mHistoryUriWithProfile = appendProfile(History.CONTENT_URI);
        mHistoryExpireUriWithProfile = appendProfile(History.CONTENT_OLD_URI);
        mCombinedUriWithProfile = appendProfile(Combined.CONTENT_URI);
        mFaviconsUriWithProfile = appendProfile(Favicons.CONTENT_URI);
        mThumbnailsUriWithProfile = appendProfile(Thumbnails.CONTENT_URI);

        mDeletedHistoryUriWithProfile = mHistoryUriWithProfile.buildUpon().
            appendQueryParameter(BrowserContract.PARAM_SHOW_DELETED, "1").build();

        mUpdateHistoryUriWithProfile = mHistoryUriWithProfile.buildUpon().
            appendQueryParameter(BrowserContract.PARAM_INCREMENT_VISITS, "true").
            appendQueryParameter(BrowserContract.PARAM_INSERT_IF_NEEDED, "true").build();
    }

    // Invalidate cached data
    @Override
    public void invalidateCachedState() {
        mDesktopBookmarksExist = null;
    }

    private Uri historyUriWithLimit(int limit) {
        return mHistoryUriWithProfile.buildUpon().appendQueryParameter(BrowserContract.PARAM_LIMIT,
                                                                       String.valueOf(limit)).build();
    }

    private Uri bookmarksUriWithLimit(int limit) {
        return mBookmarksUriWithProfile.buildUpon().appendQueryParameter(BrowserContract.PARAM_LIMIT,
                                                                         String.valueOf(limit)).build();
    }

    private Uri combinedUriWithLimit(int limit) {
        return mCombinedUriWithProfile.buildUpon().appendQueryParameter(BrowserContract.PARAM_LIMIT,
                String.valueOf(limit)).build();
    }

    private Uri appendProfile(Uri uri) {
        return uri.buildUpon().appendQueryParameter(BrowserContract.PARAM_PROFILE, mProfile).build();
    }

    private Uri getAllBookmarksUri() {
        Uri.Builder uriBuilder = mBookmarksUriWithProfile.buildUpon()
            .appendQueryParameter(BrowserContract.PARAM_SHOW_DELETED, "1");
        return uriBuilder.build();
    }

    private Uri getAllHistoryUri() {
        Uri.Builder uriBuilder = mHistoryUriWithProfile.buildUpon()
            .appendQueryParameter(BrowserContract.PARAM_SHOW_DELETED, "1");
        return uriBuilder.build();
    }

    private Uri getAllFaviconsUri() {
        Uri.Builder uriBuilder = mFaviconsUriWithProfile.buildUpon()
            .appendQueryParameter(BrowserContract.PARAM_SHOW_DELETED, "1");
        return uriBuilder.build();
    }

    private Cursor filterAllSites(ContentResolver cr, String[] projection, CharSequence constraint,
            int limit, CharSequence urlFilter) {
        return filterAllSites(cr, projection, constraint, limit, urlFilter, "", null);
    }

    private Cursor filterAllSites(ContentResolver cr, String[] projection, CharSequence constraint,
            int limit, CharSequence urlFilter, String selection, String[] selectionArgs) {
        // The combined history/bookmarks selection queries for sites with a url or title containing
        // the constraint string(s), treating space-separated words as separate constraints
        if (!TextUtils.isEmpty(constraint)) {
          String[] constraintWords = constraint.toString().split(" ");
          for (int i = 0; i < constraintWords.length; i++) {
              selection = DBUtils.concatenateWhere(selection, "(" + Combined.URL + " LIKE ? OR " +
                                                                    Combined.TITLE + " LIKE ?)");
              String constraintWord =  "%" + constraintWords[i] + "%";
              selectionArgs = DBUtils.appendSelectionArgs(selectionArgs,
                  new String[] { constraintWord, constraintWord });
          }
        }

        if (urlFilter != null) {
            selection = DBUtils.concatenateWhere(selection, "(" + Combined.URL + " NOT LIKE ?)");
            selectionArgs = DBUtils.appendSelectionArgs(selectionArgs, new String[] { urlFilter.toString() });
        }

        // Our version of frecency is computed by scaling the number of visits by a multiplier
        // that approximates Gaussian decay, based on how long ago the entry was last visited.
        // Since we're limited by the math we can do with sqlite, we're calculating this
        // approximation using the Cauchy distribution: multiplier = 15^2 / (age^2 + 15^2).
        // Using 15 as our scale parameter, we get a constant 15^2 = 225. Following this math,
        // frecencyScore = numVisits * max(1, 100 * 225 / (age*age + 225)). (See bug 704977)
        // We also give bookmarks an extra bonus boost by adding 100 points to their frecency score.
        final String sortOrder = BrowserContract.getFrecencySortOrder(true, false);

        Cursor c = cr.query(combinedUriWithLimit(limit),
                            projection,
                            selection,
                            selectionArgs,
                            sortOrder);

        return new LocalDBCursor(c);
    }

    @Override
    public int getCount(ContentResolver cr, String database) {
        int count = 0;
        String[] columns = null;
        String constraint = null;
        Uri uri = null;
        if ("history".equals(database)) {
            uri = mHistoryUriWithProfile;
            columns = new String[] { History._ID };
            constraint = Combined.VISITS + " > 0";
        } else if ("bookmarks".equals(database)) {
            uri = mBookmarksUriWithProfile;
            columns = new String[] { Bookmarks._ID };
            // ignore folders, tags, keywords, separators, etc.
            constraint = Bookmarks.TYPE + " = " + Bookmarks.TYPE_BOOKMARK;
        } else if ("thumbnails".equals(database)) {
            uri = mThumbnailsUriWithProfile;
            columns = new String[] { Thumbnails._ID };
        } else if ("favicons".equals(database)) {
            uri = mFaviconsUriWithProfile;
            columns = new String[] { Favicons._ID };
        }
        if (uri != null) {
            Cursor cursor = null;

            try {
                cursor = cr.query(uri, columns, constraint, null, null);
                count = cursor.getCount();
            } finally {
                if (cursor != null)
                    cursor.close();
            }
        }
        debug("Got count " + count + " for " + database);
        return count;
    }

    @Override
    public Cursor filter(ContentResolver cr, CharSequence constraint, int limit) {
        return filterAllSites(cr,
                              new String[] { Combined._ID,
                                             Combined.URL,
                                             Combined.TITLE,
                                             Combined.DISPLAY,
                                             Combined.BOOKMARK_ID,
                                             Combined.HISTORY_ID },
                              constraint,
                              limit,
                              null);
    }

    @Override
    public Cursor getTopSites(ContentResolver cr, int limit) {
        // Filter out bookmarks that don't have real parents (e.g. pinned sites or reading list items)
        String selection = DBUtils.concatenateWhere("", Combined.URL + " NOT IN (SELECT " +
                                             Bookmarks.URL + " FROM bookmarks WHERE " +
                                             DBUtils.qualifyColumn("bookmarks", Bookmarks.PARENT) + " < ? AND " +
                                             DBUtils.qualifyColumn("bookmarks", Bookmarks.IS_DELETED) + " == 0)");
        String[] selectionArgs = new String[] { String.valueOf(Bookmarks.FIXED_ROOT_ID) };

        return filterAllSites(cr,
                              new String[] { Combined._ID,
                                             Combined.URL,
                                             Combined.TITLE,
                                             Combined.DISPLAY,
                                             Combined.BOOKMARK_ID,
                                             Combined.HISTORY_ID },
                              "",
                              limit,
                              AboutPages.URL_FILTER,
                              selection,
                              selectionArgs);
    }

    @Override
    public void updateVisitedHistory(ContentResolver cr, String uri) {
        ContentValues values = new ContentValues();

        values.put(History.URL, uri);
        values.put(History.DATE_LAST_VISITED, System.currentTimeMillis());
        values.put(History.IS_DELETED, 0);

        // This will insert a new history entry if one for this URL
        // doesn't already exist
        cr.update(mUpdateHistoryUriWithProfile,
                  values,
                  History.URL + " = ?",
                  new String[] { uri });
    }

    @Override
    public void updateHistoryTitle(ContentResolver cr, String uri, String title) {
        ContentValues values = new ContentValues();
        values.put(History.TITLE, title);

        cr.update(mHistoryUriWithProfile,
                  values,
                  History.URL + " = ?",
                  new String[] { uri });
    }

    @Override
    public void updateHistoryEntry(ContentResolver cr, String uri, String title,
                                   long date, int visits) {
        int oldVisits = 0;
        Cursor cursor = null;
        try {
            cursor = cr.query(mHistoryUriWithProfile,
                              new String[] { History.VISITS },
                              History.URL + " = ?",
                              new String[] { uri },
                              null);

            if (cursor.moveToFirst()) {
                oldVisits = cursor.getInt(0);
            }
        } finally {
            if (cursor != null)
                cursor.close();
        }

        ContentValues values = new ContentValues();
        values.put(History.DATE_LAST_VISITED, date);
        values.put(History.VISITS, oldVisits + visits);
        if (title != null) {
            values.put(History.TITLE, title);
        }

        cr.update(mHistoryUriWithProfile,
                  values,
                  History.URL + " = ?",
                  new String[] { uri });
    }

    @Override
    public Cursor getAllVisitedHistory(ContentResolver cr) {
        Cursor c = cr.query(mHistoryUriWithProfile,
                            new String[] { History.URL },
                            History.VISITS + " > 0",
                            null,
                            null);

        return new LocalDBCursor(c);
    }

    @Override
    public Cursor getRecentHistory(ContentResolver cr, int limit) {
        Cursor c = cr.query(combinedUriWithLimit(limit),
                            new String[] { Combined._ID,
                                           Combined.BOOKMARK_ID,
                                           Combined.HISTORY_ID,
                                           Combined.URL,
                                           Combined.TITLE,
                                           Combined.DISPLAY,
                                           Combined.DATE_LAST_VISITED,
                                           Combined.VISITS },
                            History.DATE_LAST_VISITED + " > 0",
                            null,
                            History.DATE_LAST_VISITED + " DESC");

        return new LocalDBCursor(c);
    }

    @Override
    public void expireHistory(ContentResolver cr, ExpirePriority priority) {
        Uri url = mHistoryExpireUriWithProfile;
        url = url.buildUpon().appendQueryParameter(BrowserContract.PARAM_EXPIRE_PRIORITY, priority.toString()).build();
        cr.delete(url, null, null);
    }

    @Override
    public void removeHistoryEntry(ContentResolver cr, int id) {
        cr.delete(mHistoryUriWithProfile,
                  History._ID + " = ?",
                  new String[] { String.valueOf(id) });
    }

    @Override
    public void removeHistoryEntry(ContentResolver cr, String url) {
        int deleted = cr.delete(mHistoryUriWithProfile,
                  History.URL + " = ?",
                  new String[] { url });
    }

    @Override
    public void clearHistory(ContentResolver cr) {
        cr.delete(mHistoryUriWithProfile, null, null);
    }

    @Override
    public Cursor getBookmarksInFolder(ContentResolver cr, long folderId) {
        Cursor c = null;
        boolean addDesktopFolder = false;

        // We always want to show mobile bookmarks in the root view.
        if (folderId == Bookmarks.FIXED_ROOT_ID) {
            folderId = getFolderIdFromGuid(cr, Bookmarks.MOBILE_FOLDER_GUID);

            // We'll add a fake "Desktop Bookmarks" folder to the root view if desktop 
            // bookmarks exist, so that the user can still access non-mobile bookmarks.
            addDesktopFolder = desktopBookmarksExist(cr);
        }

        if (folderId == Bookmarks.FAKE_DESKTOP_FOLDER_ID) {
            // Since the "Desktop Bookmarks" folder doesn't actually exist, we
            // just fake it by querying specifically certain known desktop folders.
            c = cr.query(mBookmarksUriWithProfile,
                         DEFAULT_BOOKMARK_COLUMNS,
                         Bookmarks.GUID + " = ? OR " +
                         Bookmarks.GUID + " = ? OR " +
                         Bookmarks.GUID + " = ?",
                         new String[] { Bookmarks.TOOLBAR_FOLDER_GUID,
                                        Bookmarks.MENU_FOLDER_GUID,
                                        Bookmarks.UNFILED_FOLDER_GUID },
                         null);
        } else {
            // Right now, we only support showing folder and bookmark type of
            // entries. We should add support for other types though (bug 737024)
            c = cr.query(mBookmarksUriWithProfile,
                         DEFAULT_BOOKMARK_COLUMNS,
                         Bookmarks.PARENT + " = ? AND " +
                         "(" + Bookmarks.TYPE + " = ? OR " +
                            "(" + Bookmarks.TYPE + " = ? AND " + Bookmarks.URL + " IS NOT NULL))",
                         new String[] { String.valueOf(folderId),
                                        String.valueOf(Bookmarks.TYPE_FOLDER),
                                        String.valueOf(Bookmarks.TYPE_BOOKMARK) },
                         null);
        }

        if (addDesktopFolder) {
            // Wrap cursor to add fake desktop bookmarks and reading list folders
            c = new SpecialFoldersCursorWrapper(c, addDesktopFolder);
        }

        return new LocalDBCursor(c);
    }

    // Returns true if any desktop bookmarks exist, which will be true if the user
    // has set up sync at one point, or done a profile migration from XUL fennec.
    private boolean desktopBookmarksExist(ContentResolver cr) {
        if (mDesktopBookmarksExist != null)
            return mDesktopBookmarksExist;

        Cursor c = null;
        int count = 0;
        try {
            // Check to see if there are any bookmarks in one of our three
            // fixed "Desktop Boomarks" folders.
            c = cr.query(bookmarksUriWithLimit(1),
                         new String[] { Bookmarks._ID },
                         Bookmarks.PARENT + " = ? OR " +
                         Bookmarks.PARENT + " = ? OR " +
                         Bookmarks.PARENT + " = ?",
                         new String[] { String.valueOf(getFolderIdFromGuid(cr, Bookmarks.TOOLBAR_FOLDER_GUID)),
                                        String.valueOf(getFolderIdFromGuid(cr, Bookmarks.MENU_FOLDER_GUID)),
                                        String.valueOf(getFolderIdFromGuid(cr, Bookmarks.UNFILED_FOLDER_GUID)) },
                         null);
            count = c.getCount();
        } finally {
            if (c != null)
                c.close();
        }

        // Cache result for future queries
        mDesktopBookmarksExist = (count > 0);
        return mDesktopBookmarksExist;
    }

    @Override
    public int getReadingListCount(ContentResolver cr) {
        // This method is about the Reading List, not normal bookmarks
        Cursor c = null;
        try {
            c = cr.query(mBookmarksUriWithProfile,
                         new String[] { Bookmarks._ID },
                         Bookmarks.PARENT + " = ?",
                         new String[] { String.valueOf(Bookmarks.FIXED_READING_LIST_ID) },
                         null);
            return c.getCount();
        } finally {
            if (c != null)
                c.close();
        }
    }

    @Override
    public boolean isBookmark(ContentResolver cr, String uri) {
        // This method is about normal bookmarks, not the Reading List
        Cursor c = null;
        try {
            c = cr.query(bookmarksUriWithLimit(1),
                         new String[] { Bookmarks._ID },
                         Bookmarks.URL + " = ? AND " +
                                 Bookmarks.PARENT + " != ? AND " +
                                 Bookmarks.PARENT + " != ?",
                         new String[] { uri,
                                 String.valueOf(Bookmarks.FIXED_READING_LIST_ID),
                                 String.valueOf(Bookmarks.FIXED_PINNED_LIST_ID) },
                         Bookmarks.URL);
            return c.getCount() > 0;
        } catch (NullPointerException e) {
            Log.e(LOGTAG, "NullPointerException in isBookmark");
        } finally {
            if (c != null)
                c.close();
        }

        return false;
    }

    @Override
    public boolean isReadingListItem(ContentResolver cr, String uri) {
        Cursor c = null;
        try {
            c = cr.query(mBookmarksUriWithProfile,
                         new String[] { Bookmarks._ID },
                         Bookmarks.URL + " = ? AND " +
                         Bookmarks.PARENT + " == ?",
                         new String[] { uri,
                                        String.valueOf(Bookmarks.FIXED_READING_LIST_ID) },
                         Bookmarks.URL);

            return c.getCount() > 0;
        } catch (NullPointerException e) {
            Log.e(LOGTAG, "NullPointerException in isReadingListItem");
        } finally {
            if (c != null)
                c.close();
        }

        return false;
    }

    @Override
    public String getUrlForKeyword(ContentResolver cr, String keyword) {
        Cursor c = null;
        try {
            c = cr.query(mBookmarksUriWithProfile,
                         new String[] { Bookmarks.URL },
                         Bookmarks.KEYWORD + " = ?",
                         new String[] { keyword },
                         null);

            if (c.moveToFirst())
                return c.getString(c.getColumnIndexOrThrow(Bookmarks.URL));
        } finally {
            if (c != null)
                c.close();
        }

        return null;
    }

    private synchronized long getFolderIdFromGuid(ContentResolver cr, String guid) {
        if (mFolderIdMap.containsKey(guid))
            return mFolderIdMap.get(guid);

        long folderId = -1;
        Cursor c = null;

        try {
            c = cr.query(mBookmarksUriWithProfile,
                         new String[] { Bookmarks._ID },
                         Bookmarks.GUID + " = ?",
                         new String[] { guid },
                         null);

            if (c.moveToFirst())
                folderId = c.getLong(c.getColumnIndexOrThrow(Bookmarks._ID));
        } finally {
            if (c != null)
                c.close();
        }

        mFolderIdMap.put(guid, folderId);
        return folderId;
    }

    /**
     * Find parents of records that match the provided criteria, and bump their
     * modified timestamp.
     */
    protected void bumpParents(ContentResolver cr, String param, String value) {
        ContentValues values = new ContentValues();
        values.put(Bookmarks.DATE_MODIFIED, System.currentTimeMillis());

        String where  = param + " = ?";
        String[] args = new String[] { value };
        int updated  = cr.update(mParentsUriWithProfile, values, where, args);
        debug("Updated " + updated + " rows to new modified time.");
    }

    private void addBookmarkItem(ContentResolver cr, String title, String uri, long folderId) {
        final long now = System.currentTimeMillis();
        ContentValues values = new ContentValues();
        values.put(Browser.BookmarkColumns.TITLE, title);
        values.put(Bookmarks.URL, uri);
        values.put(Bookmarks.PARENT, folderId);
        values.put(Bookmarks.DATE_MODIFIED, now);

        // Get the page's favicon ID from the history table
        Cursor c = null;
        try {
            c = cr.query(mHistoryUriWithProfile,
                         new String[] { History.FAVICON_ID },
                         History.URL + " = ?",
                         new String[] { uri },
                         null);

            if (c.moveToFirst()) {
                int columnIndex = c.getColumnIndexOrThrow(History.FAVICON_ID);
                if (!c.isNull(columnIndex))
                    values.put(Bookmarks.FAVICON_ID, c.getLong(columnIndex));
            }
        } finally {
            if (c != null)
                c.close();
        }

        // Restore deleted record if possible
        values.put(Bookmarks.IS_DELETED, 0);

        final Uri bookmarksWithInsert = mBookmarksUriWithProfile.buildUpon()
                                          .appendQueryParameter(BrowserContract.PARAM_INSERT_IF_NEEDED, "true")
                                          .build();
        cr.update(bookmarksWithInsert,
                  values,
                  Bookmarks.URL + " = ? AND " +
                  Bookmarks.PARENT + " = " + folderId,
                  new String[] { uri });

        // Bump parent modified time using its ID.
        debug("Bumping parent modified time for addition to: " + folderId);
        final String where  = Bookmarks._ID + " = ?";
        final String[] args = new String[] { String.valueOf(folderId) };

        ContentValues bumped = new ContentValues();
        bumped.put(Bookmarks.DATE_MODIFIED, now);

        final int updated = cr.update(mBookmarksUriWithProfile, bumped, where, args);
        debug("Updated " + updated + " rows to new modified time.");
    }

    @Override
    public void addBookmark(ContentResolver cr, String title, String uri) {
        long folderId = getFolderIdFromGuid(cr, Bookmarks.MOBILE_FOLDER_GUID);
        addBookmarkItem(cr, title, uri, folderId);
    }

    @Override
    public void removeBookmark(ContentResolver cr, int id) {
        Uri contentUri = mBookmarksUriWithProfile;

        // Do this now so that the item still exists!
        final String idString = String.valueOf(id);
        bumpParents(cr, Bookmarks._ID, idString);

        final String[] idArgs = new String[] { idString };
        final String idEquals = Bookmarks._ID + " = ?";
        cr.delete(contentUri, idEquals, idArgs);
    }

    @Override
    public void removeBookmarksWithURL(ContentResolver cr, String uri) {
        Uri contentUri = mBookmarksUriWithProfile;

        // Do this now so that the items still exist!
        bumpParents(cr, Bookmarks.URL, uri);

        // Toggling bookmark on an URL should not affect the items in the reading list
        final String[] urlArgs = new String[] { uri, String.valueOf(Bookmarks.FIXED_READING_LIST_ID) };
        final String urlEquals = Bookmarks.URL + " = ? AND " + Bookmarks.PARENT + " != ?";

        cr.delete(contentUri, urlEquals, urlArgs);
    }

    @Override
    public void addReadingListItem(ContentResolver cr, String title, String uri) {
        addBookmarkItem(cr, title, uri, Bookmarks.FIXED_READING_LIST_ID);
    }

    @Override
    public void removeReadingListItemWithURL(ContentResolver cr, String uri) {
        Uri contentUri = mBookmarksUriWithProfile;

        // Do this now so that the items still exist!
        bumpParents(cr, Bookmarks.URL, uri);

        final String[] urlArgs = new String[] { uri, String.valueOf(Bookmarks.FIXED_READING_LIST_ID) };
        final String urlEquals = Bookmarks.URL + " = ? AND " + Bookmarks.PARENT + " == ?";

        cr.delete(contentUri, urlEquals, urlArgs);
    }

    @Override
    public void registerBookmarkObserver(ContentResolver cr, ContentObserver observer) {
        cr.registerContentObserver(mBookmarksUriWithProfile, false, observer);
    }

    @Override
    public void registerHistoryObserver(ContentResolver cr, ContentObserver observer) {
        cr.registerContentObserver(mHistoryUriWithProfile, false, observer);
    }

    @Override
    public void updateBookmark(ContentResolver cr, int id, String uri, String title, String keyword) {
        ContentValues values = new ContentValues();
        values.put(Browser.BookmarkColumns.TITLE, title);
        values.put(Bookmarks.URL, uri);
        values.put(Bookmarks.KEYWORD, keyword);
        values.put(Bookmarks.DATE_MODIFIED, System.currentTimeMillis());

        cr.update(mBookmarksUriWithProfile,
                  values,
                  Bookmarks._ID + " = ?",
                  new String[] { String.valueOf(id) });
    }

    /**
     * Get the favicon from the database, if any, associated with the given favicon URL. (That is,
     * the URL of the actual favicon image, not the URL of the page with which the favicon is associated.)
     * @param cr The ContentResolver to use.
     * @param faviconURL The URL of the favicon to fetch from the database.
     * @return The decoded Bitmap from the database, if any. null if none is stored.
     */
    @Override
    public Bitmap getFaviconForUrl(ContentResolver cr, String faviconURL) {
        Cursor c = null;
        byte[] b = null;

        try {
            c = cr.query(mFaviconsUriWithProfile,
                         new String[] { Favicons.DATA },
                         Favicons.URL + " = ?",
                         new String[] { faviconURL },
                         null);

            if (!c.moveToFirst()) {
                return null;
            }

            final int faviconIndex = c.getColumnIndexOrThrow(Favicons.DATA);
            b = c.getBlob(faviconIndex);
        } finally {
            if (c != null) {
                c.close();
            }
        }

        if (b == null) {
            return null;
        }

        return BitmapUtils.decodeByteArray(b);
    }

    @Override
    public String getFaviconUrlForHistoryUrl(ContentResolver cr, String uri) {
        Cursor c = null;

        try {
            c = cr.query(mHistoryUriWithProfile,
                         new String[] { History.FAVICON_URL },
                         Combined.URL + " = ?",
                         new String[] { uri },
                         null);

            if (c.moveToFirst())
                return c.getString(c.getColumnIndexOrThrow(History.FAVICON_URL));
        } finally {
            if (c != null)
                c.close();
        }

        return null;
    }

    @Override
    public void updateFaviconForUrl(ContentResolver cr, String pageUri,
            Bitmap favicon, String faviconUri) {
        ContentValues values = new ContentValues();
        values.put(Favicons.URL, faviconUri);
        values.put(Favicons.PAGE_URL, pageUri);

        byte[] data = null;
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        if (favicon.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
            data = stream.toByteArray();
        } else {
            Log.w(LOGTAG, "Favicon compression failed.");
        }
        values.put(Favicons.DATA, data);

        // Update or insert
        Uri faviconsUri = getAllFaviconsUri().buildUpon().
                appendQueryParameter(BrowserContract.PARAM_INSERT_IF_NEEDED, "true").build();

        cr.update(faviconsUri,
                  values,
                  Favicons.URL + " = ?",
                  new String[] { faviconUri });
    }

    @Override
    public void updateThumbnailForUrl(ContentResolver cr, String uri,
            BitmapDrawable thumbnail) {
        Bitmap bitmap = thumbnail.getBitmap();

        byte[] data = null;
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        if (bitmap.compress(Bitmap.CompressFormat.PNG, 0, stream)) {
            data = stream.toByteArray();
        } else {
            Log.w(LOGTAG, "Favicon compression failed.");
        }

        ContentValues values = new ContentValues();
        values.put(Thumbnails.DATA, data);
        values.put(Thumbnails.URL, uri);

        Uri thumbnailsUri = mThumbnailsUriWithProfile.buildUpon().
                appendQueryParameter(BrowserContract.PARAM_INSERT_IF_NEEDED, "true").build();
        cr.update(thumbnailsUri,
                  values,
                  Thumbnails.URL + " = ?",
                  new String[] { uri });
    }

    @Override
    public byte[] getThumbnailForUrl(ContentResolver cr, String uri) {
        Cursor c = null;
        byte[] b = null;
        try {
            c = cr.query(mThumbnailsUriWithProfile,
                         new String[]{Thumbnails.DATA},
                         Thumbnails.URL + " = ?",
                         new String[]{uri},
                         null);

            if (c.moveToFirst()) {
                int thumbnailIndex = c.getColumnIndexOrThrow(Thumbnails.DATA);
                b = c.getBlob(thumbnailIndex);
            }
        } finally {
            if (c != null)
                c.close();
        }

        return b;
    }

    /**
     * Query for non-null thumbnails matching the provided <code>urls</code>.
     * The returned cursor will have no more than, but possibly fewer than,
     * the requested number of thumbnails.
     *
     * Returns null if the provided list of URLs is empty or null.
     */
    @Override
    public Cursor getThumbnailsForUrls(ContentResolver cr, List<String> urls) {
        if (urls == null) {
            return null;
        }

        int urlCount = urls.size();
        if (urlCount == 0) {
            return null;
        }

        // Don't match against null thumbnails.
        StringBuilder selection = new StringBuilder(
                Thumbnails.DATA + " IS NOT NULL AND " +
                Thumbnails.URL + " IN ("
        );

        // Compute a (?, ?, ?) sequence to match the provided URLs.
        int i = 1;
        while (i++ < urlCount) {
            selection.append("?, ");
        }
        selection.append("?)");

        String[] selectionArgs = urls.toArray(new String[urlCount]);

        return cr.query(mThumbnailsUriWithProfile,
                        new String[] { Thumbnails.URL, Thumbnails.DATA },
                        selection.toString(),
                        selectionArgs,
                        null);
    }

    @Override
    public void removeThumbnails(ContentResolver cr) {
        cr.delete(mThumbnailsUriWithProfile, null, null);
    }

    // Utility function for updating existing history using batch operations
    public void updateHistoryInBatch(ContentResolver cr,
                                     Collection<ContentProviderOperation> operations,
                                     String url, String title,
                                     long date, int visits) {
        Cursor cursor = null;

        try {
            final String[] projection = new String[] {
                History._ID,
                History.VISITS,
                History.DATE_LAST_VISITED
            };

            // We need to get the old visit count.
            cursor = cr.query(getAllHistoryUri(),
                              projection,
                              History.URL + " = ?",
                              new String[] { url },
                              null);

            ContentValues values = new ContentValues();

            // Restore deleted record if possible
            values.put(History.IS_DELETED, 0);

            if (cursor.moveToFirst()) {
                int visitsCol = cursor.getColumnIndexOrThrow(History.VISITS);
                int dateCol = cursor.getColumnIndexOrThrow(History.DATE_LAST_VISITED);
                int oldVisits = cursor.getInt(visitsCol);
                long oldDate = cursor.getLong(dateCol);
                values.put(History.VISITS, oldVisits + visits);
                // Only update last visited if newer.
                if (date > oldDate) {
                    values.put(History.DATE_LAST_VISITED, date);
                }
            } else {
                values.put(History.VISITS, visits);
                values.put(History.DATE_LAST_VISITED, date);
            }
            if (title != null) {
                values.put(History.TITLE, title);
            }
            values.put(History.URL, url);

            Uri historyUri = getAllHistoryUri().buildUpon().
                appendQueryParameter(BrowserContract.PARAM_INSERT_IF_NEEDED, "true").build();

            // Update or insert
            ContentProviderOperation.Builder builder =
                ContentProviderOperation.newUpdate(historyUri);
            builder.withSelection(History.URL + " = ?", new String[] { url });
            builder.withValues(values);

            // Queue the operation
            operations.add(builder.build());
        } finally {
            if (cursor != null)
                cursor.close();
        }
    }

    public void updateBookmarkInBatch(ContentResolver cr,
                                      Collection<ContentProviderOperation> operations,
                                      String url, String title, String guid,
                                      long parent, long added,
                                      long modified, long position,
                                      String keyword, int type) {
        ContentValues values = new ContentValues();
        if (title == null && url != null) {
            title = url;
        }
        if (title != null) {
            values.put(Bookmarks.TITLE, title);
        }
        if (url != null) {
            values.put(Bookmarks.URL, url);
        }
        if (guid != null) {
            values.put(SyncColumns.GUID, guid);
        }
        if (keyword != null) {
            values.put(Bookmarks.KEYWORD, keyword);
        }
        if (added > 0) {
            values.put(SyncColumns.DATE_CREATED, added);
        }
        if (modified > 0) {
            values.put(SyncColumns.DATE_MODIFIED, modified);
        }
        values.put(Bookmarks.POSITION, position);
        // Restore deleted record if possible
        values.put(Bookmarks.IS_DELETED, 0);

        // This assumes no "real" folder has a negative ID. Only
        // things like the reading list folder do.
        if (parent < 0) {
            parent = getFolderIdFromGuid(cr, Bookmarks.MOBILE_FOLDER_GUID);
        }
        values.put(Bookmarks.PARENT, parent);
        values.put(Bookmarks.TYPE, type);

        Uri bookmarkUri = getAllBookmarksUri().buildUpon().
            appendQueryParameter(BrowserContract.PARAM_INSERT_IF_NEEDED, "true").build();
        // Update or insert
        ContentProviderOperation.Builder builder =
            ContentProviderOperation.newUpdate(bookmarkUri);
        if (url != null) {
            // Bookmarks are defined by their URL and Folder.
            builder.withSelection(Bookmarks.URL + " = ? AND "
                                  + Bookmarks.PARENT + " = ? AND "
                                  + Bookmarks.PARENT + " != ?",
                                  new String[] { url,
                                                 Long.toString(parent),
                                                 String.valueOf(Bookmarks.FIXED_READING_LIST_ID)
                                  });
        } else if (title != null) {
            // Or their title and parent folder. (Folders!)
            builder.withSelection(Bookmarks.TITLE + " = ? AND "
                                  + Bookmarks.PARENT + " = ? AND "
                                  + Bookmarks.PARENT + " != ?",
                                  new String[] { title,
                                                 Long.toString(parent),
                                                 String.valueOf(Bookmarks.FIXED_READING_LIST_ID)
                                  });
        } else if (type == Bookmarks.TYPE_SEPARATOR) {
            // Or their their position (seperators)
            builder.withSelection(Bookmarks.POSITION + " = ? AND "
                                  + Bookmarks.PARENT + " = ? AND "
                                  + Bookmarks.PARENT + " != ?",
                                  new String[] { Long.toString(position),
                                                 Long.toString(parent),
                                                 String.valueOf(Bookmarks.FIXED_READING_LIST_ID)
                                  });
        } else {
            Log.e(LOGTAG, "Bookmark entry without url or title and not a seperator, not added.");
        }
        builder.withValues(values);

        // Queue the operation
        operations.add(builder.build());
    }

    public void updateFaviconInBatch(ContentResolver cr,
                                     Collection<ContentProviderOperation> operations,
                                     String url, String faviconUrl,
                                     String faviconGuid, byte[] data) {
        ContentValues values = new ContentValues();
        values.put(Favicons.DATA, data);
        values.put(Favicons.PAGE_URL, url);
        if (faviconUrl != null) {
            values.put(Favicons.URL, faviconUrl);
        }

        // Update or insert
        Uri faviconsUri = getAllFaviconsUri().buildUpon().
            appendQueryParameter(BrowserContract.PARAM_INSERT_IF_NEEDED, "true").build();
        // Update or insert
        ContentProviderOperation.Builder builder =
            ContentProviderOperation.newUpdate(faviconsUri);
        builder.withValues(values);
        builder.withSelection(Favicons.PAGE_URL + " = ?", new String[] { url });
        // Queue the operation
        operations.add(builder.build());
    }

    // This wrapper adds a fake "Desktop Bookmarks" folder entry to the
    // beginning of the cursor's data set.
    private class SpecialFoldersCursorWrapper extends CursorWrapper {
        private int mIndexOffset;

        private int mDesktopBookmarksIndex = -1;

        private boolean mAtDesktopBookmarksPosition = false;

        public SpecialFoldersCursorWrapper(Cursor c, boolean showDesktopBookmarks) {
            super(c);

            mIndexOffset = 0;

            if (showDesktopBookmarks) {
                mDesktopBookmarksIndex = mIndexOffset;
                mIndexOffset++;
            }
        }

        @Override
        public int getCount() {
            return super.getCount() + mIndexOffset;
        }

        @Override
        public boolean moveToPosition(int position) {
            mAtDesktopBookmarksPosition = (mDesktopBookmarksIndex == position);

            if (mAtDesktopBookmarksPosition)
                return true;

            return super.moveToPosition(position - mIndexOffset);
        }

        @Override
        public long getLong(int columnIndex) {
            if (!mAtDesktopBookmarksPosition)
                return super.getLong(columnIndex);

            if (columnIndex == getColumnIndex(Bookmarks.PARENT)) {
                return Bookmarks.FIXED_ROOT_ID;
            }

            return -1;
        }

        @Override
        public int getInt(int columnIndex) {
            if (!mAtDesktopBookmarksPosition)
                return super.getInt(columnIndex);

            if (columnIndex == getColumnIndex(Bookmarks._ID) && mAtDesktopBookmarksPosition)
                    return Bookmarks.FAKE_DESKTOP_FOLDER_ID;

            if (columnIndex == getColumnIndex(Bookmarks.TYPE))
                return Bookmarks.TYPE_FOLDER;

            return -1;
        }

        @Override
        public String getString(int columnIndex) {
            if (!mAtDesktopBookmarksPosition)
                return super.getString(columnIndex);

            if (columnIndex == getColumnIndex(Bookmarks.GUID) && mAtDesktopBookmarksPosition)
                    return Bookmarks.FAKE_DESKTOP_FOLDER_GUID;

            return "";
        }
    }

    private static class LocalDBCursor extends CursorWrapper {
        public LocalDBCursor(Cursor c) {
            super(c);
        }

        private String translateColumnName(String columnName) {
            if (columnName.equals(BrowserDB.URLColumns.URL)) {
                columnName = URLColumns.URL;
            } else if (columnName.equals(BrowserDB.URLColumns.TITLE)) {
                columnName = URLColumns.TITLE;
            } else if (columnName.equals(BrowserDB.URLColumns.FAVICON)) {
                columnName = FaviconColumns.FAVICON;
            } else if (columnName.equals(BrowserDB.URLColumns.DATE_LAST_VISITED)) {
                columnName = History.DATE_LAST_VISITED;
            } else if (columnName.equals(BrowserDB.URLColumns.VISITS)) {
                columnName = History.VISITS;
            }

            return columnName;
        }

        @Override
        public int getColumnIndex(String columnName) {
            return super.getColumnIndex(translateColumnName(columnName));
        }

        @Override
        public int getColumnIndexOrThrow(String columnName) {
            return super.getColumnIndexOrThrow(translateColumnName(columnName));
        }
    }


    @Override
    public void pinSite(ContentResolver cr, String url, String title, int position) {
        ContentValues values = new ContentValues();
        final long now = System.currentTimeMillis();
        values.put(Bookmarks.TITLE, title);
        values.put(Bookmarks.URL, url);
        values.put(Bookmarks.PARENT, Bookmarks.FIXED_PINNED_LIST_ID);
        values.put(Bookmarks.DATE_MODIFIED, now);
        values.put(Bookmarks.POSITION, position);
        values.put(Bookmarks.IS_DELETED, 0);

        // We do an update-and-replace here without deleting any existing pins for the given URL.
        // That means if the user pins a URL, then edits another thumbnail to use the same URL,
        // we'll end up with two pins for that site. This is the intended behavior, which
        // incidentally saves us a delete query.
        Uri uri = mBookmarksUriWithProfile.buildUpon()
                .appendQueryParameter(BrowserContract.PARAM_INSERT_IF_NEEDED, "true").build();
        cr.update(uri,
                  values,
                  Bookmarks.POSITION + " = ? AND " +
                  Bookmarks.PARENT + " = ?",
                  new String[] { Integer.toString(position),
                                 String.valueOf(Bookmarks.FIXED_PINNED_LIST_ID) });
    }

    @Override
    public Cursor getPinnedSites(ContentResolver cr, int limit) {
        return cr.query(bookmarksUriWithLimit(limit),
                        new String[] { Bookmarks._ID,
                                       Bookmarks.URL,
                                       Bookmarks.TITLE,
                                       Bookmarks.POSITION },
                        Bookmarks.PARENT + " == ?",
                        new String[] { String.valueOf(Bookmarks.FIXED_PINNED_LIST_ID) },
                        Bookmarks.POSITION + " ASC");
    }

    @Override
    public void unpinSite(ContentResolver cr, int position) {
        cr.delete(mBookmarksUriWithProfile,
                  Bookmarks.PARENT + " == ? AND " + Bookmarks.POSITION + " = ?",
                  new String[] {
                      String.valueOf(Bookmarks.FIXED_PINNED_LIST_ID),
                      Integer.toString(position)
                  });
    }

    @Override
    public void unpinAllSites(ContentResolver cr) {
        cr.delete(mBookmarksUriWithProfile,
                  Bookmarks.PARENT + " == ?",
                  new String[] {
                      String.valueOf(Bookmarks.FIXED_PINNED_LIST_ID)
                  });
    }

    @Override
    public boolean isVisited(ContentResolver cr, String uri) {
        int count = 0;
        Cursor c = null;

        try {
            c = cr.query(historyUriWithLimit(1),
                         new String[] { History._ID },
                         History.URL + " = ?",
                         new String[] { uri },
                         History.URL);
            count = c.getCount();
        } catch (NullPointerException e) {
            Log.e(LOGTAG, "NullPointerException in isVisited");
        } finally {
            if (c != null)
                c.close();
        }

        return (count > 0);
    }

    public Cursor getBookmarkForUrl(ContentResolver cr, String url) {
        Cursor c = cr.query(bookmarksUriWithLimit(1),
                            new String[] { Bookmarks._ID,
                                           Bookmarks.URL,
                                           Bookmarks.TITLE,
                                           Bookmarks.KEYWORD },
                            Bookmarks.URL + " = ?",
                            new String[] { url },
                            null);

        if (c != null && c.getCount() == 0) {
            c.close();
            c = null;
        }

        return c;
    }
}
