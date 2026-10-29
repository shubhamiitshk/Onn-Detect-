package com.onndetect;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.CancellationSignal;
import android.os.ParcelFileDescriptor;
import android.os.SystemClock;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.provider.OpenableColumns;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;

/** Standalone test-package process deliberately needs only Android/Java runtime classes. */
public final class DelayedMediaProvider extends ContentProvider {
    private final AtomicInteger openedPipes = new AtomicInteger();
    @Override public boolean onCreate() { return true; }
    @Override public String getType(Uri uri) {
        String path = uri.getLastPathSegment();
        if ("chat-text".equals(path)) return "text/plain";
        if ("chat-red".equals(path) || "chat-blue".equals(path)) return "image/png";
        return "application/octet-stream";
    }
    @Override public Cursor query(Uri uri, String[] projection, String selection,
                                  String[] args, String order) {
        return query(uri, projection, selection, args, order, null);
    }
    @Override public Cursor query(Uri uri, String[] projection, String selection,
                                  String[] args, String order, CancellationSignal signal) {
        if ("delayed-query".equals(uri.getLastPathSegment())) {
            long deadline = SystemClock.elapsedRealtime() + 5_000;
            while (SystemClock.elapsedRealtime() < deadline) {
                if (signal != null) signal.throwIfCanceled();
                SystemClock.sleep(10);
            }
        }
        if ("status".equals(uri.getLastPathSegment())) {
            MatrixCursor cursor = new MatrixCursor(new String[]{"opened"});
            cursor.addRow(new Object[]{openedPipes.get()});
            return cursor;
        }
        MatrixCursor cursor = new MatrixCursor(new String[]{OpenableColumns.SIZE});
        cursor.addRow(new Object[]{null}); // Force enforcement of actual bytes rather than metadata.
        return cursor;
    }
    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        if ("chat-pack".equals(uri.getLastPathSegment())) {
            return ParcelFileDescriptor.open(new java.io.File("/data/local/tmp/onn-chat-model.zip"), ParcelFileDescriptor.MODE_READ_ONLY);
        }
        final ParcelFileDescriptor[] pipe;
        try {
            pipe = ParcelFileDescriptor.createPipe();
        } catch (IOException failure) {
            FileNotFoundException unavailable = new FileNotFoundException("Test pipe could not be opened");
            unavailable.initCause(failure);
            throw unavailable;
        }
        openedPipes.incrementAndGet();
        Thread writer = new Thread(() -> {
            try (ParcelFileDescriptor.AutoCloseOutputStream output =
                         new ParcelFileDescriptor.AutoCloseOutputStream(pipe[1])) {
                if ("chat-red".equals(uri.getLastPathSegment()) || "chat-blue".equals(uri.getLastPathSegment())) {
                    Bitmap bitmap = Bitmap.createBitmap(96, 96, Bitmap.Config.ARGB_8888);
                    try {
                        bitmap.eraseColor("chat-red".equals(uri.getLastPathSegment()) ? Color.RED : Color.BLUE);
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, output);
                    } finally { bitmap.recycle(); }
                }
                else if ("chat-text".equals(uri.getLastPathSegment())) {
                    output.write("Invoice total: ₹250. Payment is pending. This document is a test fixture.".getBytes(StandardCharsets.UTF_8));
                }
                else if ("blocked-pipe".equals(uri.getLastPathSegment())) SystemClock.sleep(3_000);
                else {
                    byte[] bytes = new byte[8 * 1024];
                    Arrays.fill(bytes, (byte) 7);
                    output.write(bytes);
                }
            } catch (IOException expectedOnCancel) { /* Importer may close the read end. */ }
        }, "ONN-test-provider");
        writer.setDaemon(true);
        writer.start();
        return pipe[0];
    }
    @Override public Uri insert(Uri uri, ContentValues values) { return null; }
    @Override public int delete(Uri uri, String selection, String[] args) { return 0; }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] args) { return 0; }
}
