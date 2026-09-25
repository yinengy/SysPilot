#define _GNU_SOURCE
#include <jni.h>
#include <dlfcn.h>
#include <stdio.h>

/* No BLAS linkage, pool setters or operations: inspect the existing JVM mapping. */
JNIEXPORT jobjectArray JNICALL
Java_org_libcontentbench_BlasObserver_snapshot(JNIEnv *env, jclass unused) {
    (void)unused;
    int (*threads)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "openblas_get_num_threads");
    int (*parallel)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "openblas_get_parallel");
    char *(*config)(void) = (char *(*)(void))dlsym(RTLD_DEFAULT, "openblas_get_config");
    char *(*core)(void) = (char *(*)(void))dlsym(RTLD_DEFAULT, "openblas_get_corename");
    Dl_info location;
    if (!threads || !parallel || !config || !core || !dladdr((void *)threads, &location)) {
        (*env)->ThrowNew(env, (*env)->FindClass(env, "java/lang/IllegalStateException"),
                        "Effective OpenBLAS metadata symbols unavailable in this JVM");
        return NULL;
    }
    char n[32], mode[32];
    snprintf(n, sizeof(n), "%d", threads());
    snprintf(mode, sizeof(mode), "%d", parallel());
    const char *values[] = {n, mode, config(), core(), location.dli_fname};
    jobjectArray result = (*env)->NewObjectArray(env, 5,
        (*env)->FindClass(env, "java/lang/String"), NULL);
    if (!result) return NULL;
    for (int i = 0; i < 5; i++) {
        jstring value = (*env)->NewStringUTF(env, values[i] ? values[i] : "");
        if (!value) return NULL;
        (*env)->SetObjectArrayElement(env, result, i, value);
        (*env)->DeleteLocalRef(env, value);
    }
    return result;
}
