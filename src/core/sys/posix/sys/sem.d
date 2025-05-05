/**
 * D header file for POSIX.
 *
 * Copyright: Copyright Sean Kelly 2005 - 2009.
 * License:   $(HTTP www.boost.org/LICENSE_1_0.txt, Boost License 1.0).
 * Authors:   Sean Kelly, Cline
 * Standards: The Open Group Base Specifications Issue 6, IEEE Std 1003.1, 2004 Edition
 */

/*          Copyright Sean Kelly 2005 - 2009.
 * Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 */
module core.sys.posix.sys.sem;

import core.sys.posix.config;
import core.sys.posix.sys.types;
import core.sys.posix.time;

version (Posix):
extern (C) nothrow @nogc:

// POSIX semaphore structure (opaque)
struct sem_t;

// Error value returned by sem_open
enum SEM_FAILED = cast(sem_t*)null;

// Function declarations for POSIX semaphores
int sem_init(sem_t* sem, int pshared, uint value);
int sem_destroy(sem_t* sem);
sem_t* sem_open(const(char)* name, int oflag, ...);
int sem_close(sem_t* sem);
int sem_unlink(const(char)* name);
int sem_wait(sem_t* sem);
int sem_trywait(sem_t* sem);
int sem_post(sem_t* sem);
int sem_getvalue(sem_t* sem, int* value);
