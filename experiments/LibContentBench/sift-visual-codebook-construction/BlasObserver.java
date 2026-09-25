package org.libcontentbench;

/** Read-only native metadata from the BLAS already loaded by Spark/netlib. */
public final class BlasObserver {
    public static native String[] snapshot();
    private BlasObserver() { }
}
