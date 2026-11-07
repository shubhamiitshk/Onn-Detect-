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
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;
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
        if ("chat-video".equals(path)) return "video/mp4";
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
        MatrixCursor cursor = new MatrixCursor(new String[]{OpenableColumns.SIZE, OpenableColumns.DISPLAY_NAME});
        String displayName = "legacy-chat-pack".equals(uri.getLastPathSegment()) ? "qwen35-0.8b-v1.zip" :
                "chat-pack".equals(uri.getLastPathSegment()) ? "qwen35-0.8b-base-chat.zip" : "selected-model-file";
        cursor.addRow(new Object[]{null, displayName}); // Force enforcement of actual bytes rather than metadata.
        return cursor;
    }
    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        if ("chat-pack".equals(uri.getLastPathSegment())) {
            return ParcelFileDescriptor.open(new java.io.File("/data/local/tmp/onn-chat-model.zip"), ParcelFileDescriptor.MODE_READ_ONLY);
        }
        if ("legacy-chat-pack".equals(uri.getLastPathSegment())) {
            return ParcelFileDescriptor.open(new java.io.File("/data/local/tmp/onn-legacy-chat-model.zip"), ParcelFileDescriptor.MODE_READ_ONLY);
        }
        if ("chat-video".equals(uri.getLastPathSegment())) {
            try { return ParcelFileDescriptor.open(createBlueVideo(), ParcelFileDescriptor.MODE_READ_ONLY); }
            catch (Exception failure) { throw new FileNotFoundException("Could not generate video fixture: " + failure); }
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
                if ("bad-chat-pack".equals(uri.getLastPathSegment())) {
                    java.util.zip.ZipOutputStream zip = new java.util.zip.ZipOutputStream(output);
                    zip.putNextEntry(new java.util.zip.ZipEntry("chat_model.json"));
                    zip.write("{\"schema_version\":999,\"purpose\":\"conversation\"}".getBytes(StandardCharsets.UTF_8));
                    zip.closeEntry(); zip.finish();
                }
                else if ("chat-red".equals(uri.getLastPathSegment()) || "chat-blue".equals(uri.getLastPathSegment())) {
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

    /** Real AVC/MP4 bytes exercise the production decoder; colour is not revealed in the filename. */
    private synchronized java.io.File createBlueVideo() throws IOException {
        java.io.File file = new java.io.File(getContext().getCacheDir(), "chat-video-fixture.mp4");
        if (file.isFile() && file.length() > 0) return file;
        MediaCodec codec = MediaCodec.createEncoderByType("video/avc");
        MediaMuxer muxer = new MediaMuxer(file.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
        boolean muxerStarted = false;
        try {
            MediaFormat format = MediaFormat.createVideoFormat("video/avc", 128, 128);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            format.setInteger(MediaFormat.KEY_BIT_RATE, 64000);
            format.setInteger(MediaFormat.KEY_FRAME_RATE, 4);
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
            format.setInteger(MediaFormat.KEY_COLOR_STANDARD, MediaFormat.COLOR_STANDARD_BT601_PAL);
            format.setInteger(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED);
            format.setInteger(MediaFormat.KEY_COLOR_TRANSFER, MediaFormat.COLOR_TRANSFER_SDR_VIDEO);
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            codec.start();
            byte[] frame = new byte[128 * 128 * 3 / 2];
            Arrays.fill(frame, 0, 128 * 128, (byte) 41);
            Arrays.fill(frame, 128 * 128, 128 * 128 * 5 / 4, (byte) 240);
            Arrays.fill(frame, 128 * 128 * 5 / 4, frame.length, (byte) 110);
            int submitted = 0;
            int track = -1;
            boolean endedInput = false;
            boolean endedOutput = false;
            long deadline = SystemClock.elapsedRealtime() + 15000;
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            while (!endedOutput && SystemClock.elapsedRealtime() < deadline) {
                if (!endedInput) {
                    int input = codec.dequeueInputBuffer(10000);
                    if (input >= 0) {
                        if (submitted < 12) {
                            java.nio.ByteBuffer buffer = codec.getInputBuffer(input);
                            buffer.clear(); buffer.put(frame);
                            codec.queueInputBuffer(input, 0, frame.length, submitted * 250000L, 0);
                            submitted++;
                        } else {
                            codec.queueInputBuffer(input, 0, 0, 3000000L, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            endedInput = true;
                        }
                    }
                }
                int output = codec.dequeueOutputBuffer(info, 10000);
                if (output == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    track = muxer.addTrack(codec.getOutputFormat()); muxer.start(); muxerStarted = true;
                } else if (output >= 0) {
                    try {
                        if (info.size > 0 && (info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0) {
                            if (!muxerStarted) throw new IOException("Encoder output arrived before its format");
                            java.nio.ByteBuffer buffer = codec.getOutputBuffer(output);
                            buffer.position(info.offset); buffer.limit(info.offset + info.size);
                            muxer.writeSampleData(track, buffer, info);
                        }
                        endedOutput = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                    } finally { codec.releaseOutputBuffer(output, false); }
                }
            }
            if (!endedOutput) throw new IOException("Video encoding deadline exceeded");
        } finally {
            try { codec.stop(); } catch (Exception ignored) { }
            codec.release();
            if (muxerStarted) muxer.stop();
            muxer.release();
        }
        return file;
    }
}
