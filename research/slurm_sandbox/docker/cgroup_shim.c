#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdarg.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#define PROD "/sys/fs/cgroup"
#define FAKE_CGFS "/tmp/cgfs"
#define ROOT_LINE "0::/\n"
#define STEP_LINE "0::/system.slice/slurmnode_slurmstepd.scope/system\n"

static const char *stepd_content = "0::/system.slice/slurmnode_slurmstepd.scope/system\n";

static FILE *real_fopen_impl(const char *path, const char *mode)
{
	static FILE *(*real)(const char *, const char *);
	if (!real)
		real = dlsym(RTLD_NEXT, "fopen");
	return real(path, mode);
}

static void write_file(const char *path, const char *content)
{
	FILE *f = real_fopen_impl(path, "w");
	if (f) {
		fputs(content, f);
		fclose(f);
	}
}

static void read_comm(pid_t pid, char *out, size_t n)
{
	char p[64];
	out[0] = '\0';
	snprintf(p, sizeof(p), "/proc/%d/comm", (int) pid);
	FILE *f = real_fopen_impl(p, "r");
	if (f) {
		if (fgets(out, (int) n, f)) {
			size_t l = strlen(out);
			while (l && (out[l-1] == '\n' || out[l-1] == '\r'))
				out[--l] = '\0';
		}
		fclose(f);
	}
}

/* Decide the fake /proc/<pid>/cgroup content for a given process. */
static const char *content_for(pid_t pid)
{
	char comm[64];
	if (pid == 1)
		return ROOT_LINE;
	read_comm(pid, comm, sizeof(comm));
	if (strcmp(comm, "slurmstepd") == 0)
		return stepd_content;
	return ROOT_LINE;
}

static void ensure_fake(pid_t pid)
{
	char p[64];
	char cur[256];
	FILE *f;
	snprintf(p, sizeof(p), "/tmp/fake_cgroup.%d", (int) pid);
	f = real_fopen_impl(p, "r");
	if (f) {
		size_t n = fread(cur, 1, sizeof(cur) - 1, f);
		fclose(f);
		cur[n] = '\0';
		if (strcmp(cur, content_for(pid)) == 0)
			return;
	}
	write_file(p, content_for(pid));
}

/* Returns 1 and sets pid if path is a cgroup membership file. */
static int is_proc_cgroup(const char *path, pid_t *pid)
{
	if (!path)
		return 0;
	if (strcmp(path, "/proc/self/cgroup") == 0) {
		*pid = getpid();
		return 1;
	}
	if (strncmp(path, "/proc/", 6) == 0) {
		const char *q = path + 6;
		long v = 0;
		if (isdigit((unsigned char)*q)) {
			while (isdigit((unsigned char)*q))
				v = v * 10 + (*q++ - '0');
			if (strcmp(q, "/cgroup") == 0) {
				*pid = (pid_t) v;
				return 1;
			}
		}
	}
	return 0;
}

static char map_buf[4096];

static const char *map_path(const char *path)
{
	pid_t cpid;
	if (!path)
		return path;
	if (is_proc_cgroup(path, &cpid)) {
		ensure_fake(cpid);
		snprintf(map_buf, sizeof(map_buf), "/tmp/fake_cgroup.%d", (int) cpid);
		return map_buf;
	}
	if (strcmp(path, PROD) == 0)
		return FAKE_CGFS;
	if (strncmp(path, PROD "/", 14) == 0) {
		snprintf(map_buf, sizeof(map_buf), "%s%s", FAKE_CGFS, path + strlen(PROD));
		return map_buf;
	}
	return path;
}

FILE *real_fopen(const char *path, const char *mode)
{
	return real_fopen_impl(path, mode);
}

FILE *fopen(const char *path, const char *mode)
{
	return real_fopen_impl(map_path(path), mode);
}

FILE *fopen64(const char *path, const char *mode)
{
	return fopen(path, mode);
}

int open(const char *path, int flags, ...)
{
	static int (*real)(const char *, int, ...) = NULL;
	mode_t mode = 0;
	if (!real)
		real = dlsym(RTLD_NEXT, "open");
	if (flags & O_CREAT) {
		va_list ap;
		va_start(ap, flags);
		mode = va_arg(ap, mode_t);
		va_end(ap);
	}
	return real(map_path(path), flags, mode);
}

int open64(const char *path, int flags, ...)
{
	return open(path, flags);
}

int openat(int dirfd, const char *path, int flags, ...)
{
	static int (*real)(int, const char *, int, ...) = NULL;
	mode_t mode = 0;
	if (!real)
		real = dlsym(RTLD_NEXT, "openat");
	if (flags & O_CREAT) {
		va_list ap;
		va_start(ap, flags);
		mode = va_arg(ap, mode_t);
		va_end(ap);
	}
	if (path && path[0] == '/')
		return real(dirfd, map_path(path), flags, mode);
	return real(dirfd, path, flags, mode);
}

static void populate_dir(const char *dir)
{
	static const char *empty[] = {
		"cgroup.procs", "cgroup.subtree_control", "cgroup.type",
		"cgroup.stat", "cgroup.events", "cgroup.threads",
		"cgroup.cpus.controllers", "cgroup.cpus.subtree_control",
		"cgroup.memory.controllers", "cgroup.memory.subtree_control",
		"cgroup.freeze", NULL
	};
	char p[4096];
	int i;
	struct stat st;
	for (i = 0; empty[i]; i++) {
		snprintf(p, sizeof(p), "%s/%s", dir, empty[i]);
		if (stat(p, &st) != 0) {
			FILE *f = real_fopen_impl(p, "w");
			if (f)
				fclose(f);
		}
	}
	snprintf(p, sizeof(p), "%s/cgroup.controllers", dir);
	if (stat(p, &st) != 0) {
		FILE *f = real_fopen_impl(p, "w");
		if (f) {
			fputs("cpuset cpu memory pids\n", f);
			fclose(f);
		}
	}
}

int mkdir(const char *path, mode_t mode)
{
	static int (*real)(const char *, mode_t);
	if (!real)
		real = dlsym(RTLD_NEXT, "mkdir");
	const char *mp = map_path(path);
	int rc = real(mp, mode);
	if (rc == 0 && strncmp(mp, FAKE_CGFS, strlen(FAKE_CGFS)) == 0)
		populate_dir(mp);
	return rc;
}

int stat(const char *path, struct stat *st)
{
	static int (*real)(const char *, struct stat *);
	if (!real)
		real = dlsym(RTLD_NEXT, "stat");
	return real(map_path(path), st);
}

int lstat(const char *path, struct stat *st)
{
	static int (*real)(const char *, struct stat *);
	if (!real)
		real = dlsym(RTLD_NEXT, "lstat");
	return real(map_path(path), st);
}

int access(const char *path, int amode)
{
	static int (*real)(const char *, int);
	if (!real)
		real = dlsym(RTLD_NEXT, "access");
	return real(map_path(path), amode);
}

int unlink(const char *path)
{
	static int (*real)(const char *);
	if (!real)
		real = dlsym(RTLD_NEXT, "unlink");
	return real(map_path(path));
}

int rmdir(const char *path)
{
	static int (*real)(const char *);
	if (!real)
		real = dlsym(RTLD_NEXT, "rmdir");
	return real(map_path(path));
}
