## Overview

#### Process and Threads

Process, a program in execution, is an instance of a running program.

Process Memory Layout
    - Code (Text): Program instructions. Ptogram language to machine language. read-only
    - Data: Global and static variables. All threads share the data.
    - Heap: Dynamically allocated memory (malloc, new). All threads share the heap.
    - Stack: geçici variables, function calls and return addresses and every thread has its own stack

Threads :
    - All threads has own stack So each thread work indipendently
    - The operating system allocates a specific time slice to each thread (time slicing).

Concurrency = executing tasks in short intervals and advancing them together.

Parallelism = Parallelism refers to the **actual simultaneous execution of multiple tasks**. A task is executed across **multiple cores or processors** at the same time.

Shared Memory (Symmetric Multiprocessing) SMP
    - All threads share the same memory
    - Threads communicate by reading and writing to the same memory locations
    - All cores have equal access to memory
    - Threads can be synchronized using locks, barriers, etc.

  

NUMA (Non-Uniform Memory Access)
    - Each core has its own memory
    - Cores can access their own memory faster than other cores' memory
    - Threads can be pinned to cores to improve performance-
    -OpenMp using Numa



```C++
#include <cstdio>  
#include <omp.h>  
  
int main(){  
    #pragma omp parallel  
    {  
        int ID = 0;  
        printf("Hello World\n");  
  
    }  
  
}

```

Note: To make OpenMP work in the project, the following lines should be added to the CMake.txt file.

```Cmake
find_package(OpenMP)  
  
if(OPENMP_FOUND)  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  
  
list( APPEND CUDA_NVCC_FLAGS  -Xcompiler /openmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP )
```


Multi Threading Program :

tüm program bir main thread ile başlar bu threadin ID si 0 dır.

```Cpp
#include <cstdio>  
#include <omp.h>  
  
int main(){  
    double A[1000];  
    omp_set_num_threads(4);  
    #pragma omp parallel  
    {  
        int ID = omp_get_thread_num();  
        pooh(ID,A);  
    }  
    printf("all done \n");  
  
}
```


``` Cpp
#include <cstdio>  
#include <omp.h>  
  
#define NUM_THREADS 2  
  
static long num_steps = 100000;  
double step;  
  
int main() {  
    int i = 0;  
    int nthreads = 0;  
    double pi = 0.0;  
    double sum[NUM_THREADS] = {0.0}; // Initialize sum array  
  
    step = 1.0 / (double) num_steps;  
    omp_set_num_threads(NUM_THREADS);  
  
#pragma omp parallel  
    {  
        int id = omp_get_thread_num();      //  o anki thread idyi verir  
        int nthrds = omp_get_num_threads(); // toplam threads sayısını veriri  
        double x;  
  
        if (id == 0) {  
            nthreads = nthrds;  // thredler toplam kaçtane hesapkama yaptı
        }  
  
        sum[id] = 0.0; // her bir thread kendi başlagıc toplami ile başlar  
        for (int i = id; i < num_steps; i += nthrds) {  
            x = (i + 0.5) * step;  
            sum[id] += 4.0 / (1.0 + x * x);  
        }  
    }  
  
    // Compute final value of pi  
    for (int i = 0; i < nthreads; i++) {  
        pi += sum[i] * step;  
    }  
  
    printf("Calculated PI: %lf\n", pi);  
  
    return 0;  
}
```

PAD for false sharing algorithm:

```Cpp
#include <cstdio>  
#include <omp.h>  
  
#define NUM_THREADS 2  
#define PAD 8 // assume 64 byte L1 cache line size  
  
static long num_steps = 100000;  
double step;  
  
int main() {  
    int i = 0;  
    int nthreads = 0;  
    double pi = 0.0;  
    double sum[NUM_THREADS][PAD] = {0.0}; // Initialize sum array  
  
    step = 1.0 / (double) num_steps;  
    omp_set_num_threads(NUM_THREADS);  
  
#pragma omp parallel  
    {  
        int id = omp_get_thread_num();      //  o anki thread idyi verir  
        int nthrds = omp_get_num_threads(); // toplam threads sayısını veriri  
        double x;  
  
        if (id == 0) {  
            nthreads = nthrds;  // thredler toplam kaçtane hesapkama yaptı  
        }  
  
        //sum[id] = 0.0; // her bir thread kendi başlagıc toplami ile başlar  
        for (int i = id; i < num_steps; i += nthrds) {  
            x = (i + 0.5) * step;  
            sum[id][0] += 4.0 / (1.0 + x * x);  
        }  
    }  
  
    // Compute final value of pi  
    for (int i = 0; i < nthreads; i++) {  
        pi += sum[i][0] * step;  
    }  
  
    printf("Calculated PI: %lf\n", pi);  
  
    return 0;  
}
```

## Sections and Loop task sharing

In the SPMD (Single Program Multiple Data) model, all threads run the same code, but each processes different data, provides data parallelism, works independently, and may require synchronization or data sharing.

In the SPMD model, each thread repeatedly runs the same code, but sometimes you may want to take a single task and share it among threads. This is called **task sharing**, and it is a very important feature of OpenMP. We will discuss this now.

OpenMP provides several task sharing constructs; there is a task sharing loop construct, "sections" and "section" constructs, the "single" construct, and also "tasks," which technically are not task sharing but function similarly. 

Most common task sharing construct, the loop task sharing construct.

#### Loop task sharing


```Cpp
#pragma omp parallel
#pragma omp for
	for(i=0;i<N;i++) {
	a[i] = a[i] + b [i];
	}
```


#### The Schedule

the schedule directive defines how the workload of a parallel loop should be distributed among the threads.
    -static : each thread gets a chunk of iterations .The workload is divided into fixed blocks -and assigned to each thread beforehand.
    -dynamic :This places the loop iterations into logical task units, and one thread moves on to the next iteration after finishing the current one. The difference here is that the workload sharing is done at runtime, not compile time. If there are significant differences between the threads' workloads, dynamic scheduling is more effective.
    -auto : the compiler decides the schedule
    -runtime : the schedule is decided at runtime

  
```Cpp
    #pragma omp parallel for schedule(static, 4)
    for(int i=0; i<10; i++)
    {

        // code block
    }
```



## Reduction

the reduction clause can be used to specify that a variable should be reduced across the threads involved in a parallel region. For example:

```cpp
#pragma omp parallel for reduction(+:sum) 
for (int i = 0; i < n; i++) 
{     
sum += arr[i]; 
}
```




![[Pasted image 20250305150304.png]]



## Master section nowait

```cpp
#pragma omp nowait
{
    // code block // ensures that the next block of code is not waited for
}
```

```cpp
#pragma omp sections

{
    #pragma omp section
    {
        // code block //
    }
    #pragma omp section
    {
        // code block //
    }
}
```

```cpp
//master thread is the thread that has thread id 0
//and this code block is executed by the master thread only

#pragma omp master
{
    // code block // ensures that the next block of code is executed by the master thread
}
```

```cpp
#pragma omp single
{
    // code block // ensures that the next block of code is executed by a single thread

}
```

## Synchronization.

Race condition refers to a situation where one thread is writing to a variable while another thread tries to read that variable.

Race conditions cause a program to produce different results every time it is run. To prevent race conditions, it is necessary to organize and control access to shared variables. This is done using synchronization. However, excessive synchronization negatively impacts performance, so synchronization should be minimized as much as possible.


Barrier Synncho :
	Each of thread wait for other threadto contunue

```Cpp
int main() {  
#pragma omp parallel  
    {  
        int id  = omp_get_thread_num();
        A[id] = big_cal1(id);
#pragma omp barrier // herkes A yı hesaplayana kadar bekle sonra B yi hesapla
		B[id] = big_calc2(id,A):  
         
    }  
  
    return 0;  
}

```

Mutual Exlusion:
sadece bir threadin çalışabileceği noktalar belirleyerek yapılır

 Ciritical
 same time only one thread can run this block of code other threads wait for this block of code race condition is prevented but performance can be affected

```Cpp
int main() {  
#pragma omp parallel  
    {  
        float B;
        init i ;
        init id;
        int nthrds;
        nthrds = omp_get_num_threads()
        for(i=id;i<niters;i+=nthrds){
	        B = big_job(i)
        }
 
#pragma omp critical 
		B[id] = big_calc2(id,A):  
         
    }  
  
    return 0;  
}


```



## Lock Rutiens

When you need a lower-level structure to have more control over how synchronization is performed, this is where locks come into play.

A lock is the lowest-level mutual exclusion synchronization in concurrent programming. Essentially, if you hold a lock, you are happy and can proceed with your work. However, if you try to acquire a lock while someone else is holding it, you must wait. Since locks are defined as a function API and implemented as variables, they allow you to manage many low-level details that you cannot handle with critical sections

`omp_init_lock` start a lock veriable
`omp_set_lock` open a lock 
`omp_unset_lock` when you work with done you can use this function for relase lock
`omp_destroy_lock` for free space in memory you have tu use and destroy lock 

```cpp
#include <omp.h>

int main() {
    int shared_var = 0;
    omp_lock_t lock; // Lock değişkenini tanımlıyoruz

    omp_init_lock(&lock); // Lock değişkenini başlatıyoruz

    #pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();

        omp_set_lock(&lock); // Lock açılıyor
        shared_var += 1;
        omp_unset_lock(&lock); // İşlem bitti, lock serbest bırakılıyor
    }

    omp_destroy_lock(&lock); // Kullanım bittiğinde lock'u yok ediyoruz

    return 0;
}

```



## Enviroment Variables

using the following environment variables, we can control the behavior of the OpenMP library



OMP_NUM_THREADS
    how many threads to use defaul
    export OMP_NUM_THREADS=4 // sh
    omp_set_num_threads(4); // c++

OMP_STACKSIZE
    how much stack space to allocate for each thread
    export OMP_STACKSIZE=1G // sh
    omp_set_stacksize(512 * 1024 * 1024);  // 512MB

OMP_WAIT_POLICY
    how to wait for threads to finish
    export OMP_WAIT_POLICY=active // sh
    omp_set_wait_policy(omp_active_wait); // c++
    
   passive: sleep while waiting
   pasiive to active: sleep while waiting, then spin and is very chhep

OMP_PROC_BIND
   how to bind threads to processors
   export OMP_PROC_BIND=close // sh
   omp_set_proc_bind(omp_proc_bind_close); // c++

   close: bind threads to processors close to each other
   spread: bind threads to processors far from each other
   master: bind threads to the processor of the master thread
   proc: bind threads to the processor of the thread that created them

## Data Enveriment

  

OpenMp is:

shared memory programming model so all threads can access the same memory most of the veriables in heap memory
private veriables are private to each thread So Heap is shared and Stack is private

##### Change Storage Attributes

shared : all threads share the same variable

private : each thread has its private variables its not initilaized variables

```cpp
void main()
{
	int x;
	#pragma omp parallel for private(x)
	for(int i=0; i<10; i++)
	{

		x = i;
	}
}
```

firstprivate : creat a private copy but will initilaize it to the global copy when you get to thr end of construct, the private copy is destroyed

```cpp
void main()
{
	int x = 10;
	#pragma omp parallel for firstprivate(x)
	for(int i=0; i<10; i++)
	{

		x = i;
	}
}
```

 
last pirivate : the last value of the variable is copied to the global copy

```cpp
 void main()
{
	int x = 10;
	#pragma omp parallel for lastprivate(x)
	for(int i=0; i<10; i++)
	{
	
		x = i;
	}
}

```


default(none) : you must specify the storage attributes for all variables

```cpp
void main()
{
	int x = 10;
	#pragma omp parallel for default(none) private(x)
	for(int i=0; i<10; i++)
	{
		x = i;
	}
}
```

  Why Use?
        - Forces variable scoping, prevents ambiguity.
        - Prevents sharing of variables that should not be shared by mistake.
        - Increases readability and security of the code.

  
 General Rule:
        If you want to manually specify all variables, use default(none).
        If you want all variables to be shared by default, use default(shared).

## Task

##### Internal ccontrol variables

OpenMP creates a set of threads, and information such as how many threads will be created is stored in internal control variables.

You cannot directly see these internal control variables; they are managed at runtime by the program. However, we can modify them using OpenMP runtime library functions.

The important point here is this: Internal control variables are associated with tasks, not threads.


##### **`task` directive** 
defines **independent operations (tasks)** that can be executed. The OpenMP runtime distributes these tasks among available threads in the thread pool.

Tasks are created dynamically using **`#pragma omp task`**.
Unlike static work distribution, tasks are assigned to threads **at runtime**.
**`#pragma omp taskwait`** ensures that execution waits until a task is completed.

##### Examples:

###### (divide and conquer)

```Cpp
int fib(int n) {

    int x, y;

    if (n < 2) return n;

    #pragma omp task shared(x)

    x = fib(n-1);

    #pragma omp task shared(y) // share sayesinde x ve y nin degerleri tutulabildi

    y = fib(n-2);

    #pragma omp taskwait

    return x + y;

}

  

int main() {

    int e;

    #pragma omp parallel

    {

        #pragma omp single

        for (e=ml->first; e;e->next) {

            #pragma omp task firstprivate(e)

            proses(e);

        }

    }

}

  
```


Note : date race is not a problem for rhis code because

Each time I create a task, it will take the current

 value of the e variable and use that value

 to create a private variable.

 This private variable will be a copy of the

 e pointer and will be initialized to that pointer.

  
  
##### Another example code :  

![[Pasted image 20250305122933.png]]



Assigning a Single Thread: This thread initializes a pointer to the head of the linked list.

##### Traversing the Linked List:

A while loop iterates through the list.

In each iteration, a task is created for the current node.

The pointer advances to the next node.

##### Distributing Tasks to Threads:

The single thread executing the loop generates all the tasks.

It then reaches a barrier and joins the other threads to execute the tasks.

  
##### Advantage of Parallel Execution:

If only one thread were used, all tasks would be processed sequentially.

With multiple threads, one creates the tasks while others execute them.

This significantly reduces processing time.

--------------------------------------------------------------------------

## Shared memory and Flush

In a shared-memory system, there are multiple processor units, each with its own cache. This can lead to different values for the same data across different processors. For example, while processor 3 may have a value of a variable in its cache, the value in DRAM might be different. In this case, it is important to know which value will be seen.

The compiler can optimize the order of read and write operations when compiling the code. For example, it can group scattered read and write operations together to make them more efficient for the processor. These optimizations aim to improve the performance of the code. However, when the code is executed, the order may change due to processor scheduling and interrupts.

Therefore, in shared-memory systems, memory consistency and processor cache management are quite complex, and ensuring proper data synchronization is crucial.

The flush command ensures that a thread's data is updated and properly synchronized with shared memory. It is usually necessary to call the flush command before performing a write operation. This way, data consistency is maintained.

```
#pragma omp parallel { 
int x = 0; 
#pragma omp flush(x) 
x = 10; 
pragma omp flush(x) 
 }
```

## Subsriber and producer Spinlock 

###### Spinlock (Döngüsel Kilitleme)

```Cpp
/*
**  PROGRAM: A simple serial producer/consumer program
**
**  One function generates (i.e. produces) an array of random values.  
**  A second functions consumes that array and sums it.
**
**  HISTORY: Written by Tim Mattson, April 2007.
*/
#include <omp.h>
#ifdef APPLE
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include <stdio.h>

#define N        10000

/* Some random number constants from numerical recipies */
#define SEED       2531
#define RAND_MULT  1366
#define RAND_ADD   150889
#define RAND_MOD   714025
int randy = SEED;

/* function to fill an array with random numbers */
void fill_rand(int length, double *a)
{
   int i; 
   for (i=0;i<length;i++) {
     randy = (RAND_MULT * randy + RAND_ADD) % RAND_MOD;
     *(a+i) = ((double) randy)/((double) RAND_MOD);
   }   
}

/* function to sum the elements of an array */
double Sum_array(int length, double *a)
{
   int i;  double sum = 0.0;
   for (i=0;i<length;i++)  sum += *(a+i);  
   return sum; 
}
  
int main()
{
  double *A, sum, runtime;
  int flag = 0;

  A = (double *)malloc(N*sizeof(double));

  runtime = omp_get_wtime();

  fill_rand(N, A);        // Producer: fill an array of data

  sum = Sum_array(N, A);  // Consumer: sum the array
   
  runtime = omp_get_wtime() - runtime;

  printf(" In %f seconds, The sum is %f \n",runtime,sum);
}
 
```

This, is quite simple. One function will fill an array with random values, and the other function will sum them up.

You need to wait for the array to be filled before summing it up. Here, pairwise synchronization comes .  OpenMP does not have a built-in binary synchronization mechanism.

```Cpp
 
int main()
{
  int numthreads;
  double *A, sum, runtime;
  int flag = 0;
  A = (double *)malloc(N*sizeof(double));
#pragma omp parallel sections
{
  #pragma omp section
  {
  runtime = omp_get_wtime();
  fill_rand(N, A);        // Producer: fill an array of data
  #pragma omp flush // other threads can see the array
  #pragma atomic write // for race-free
  flag = 1;
  #pragma omp flush(flag) // its meaning okey you can get data
  }
  #pragma omp section
  {
   #pragma omp flush(flag)
   while(1)
   {
	   #pragma omp flush(flag) // its for every time loking
	   #pragma omp atomic write // for race-free
		   flg_tmp = flag;
		if(flg_tmp==1)break;
   }
   #pragma omp flush
   sum = Sum_array(N, A); // Consumer: sum the array
  }
    
  runtime = omp_get_wtime() - runtime;

  printf(" In %f seconds, The sum is %f \n",runtime,sum);
}
}
 
```


## Thread Private


A threadprivate variable is a variable that is private to each thread, but all functions in the same file can access the variable of that thread.

###### Why is **thread private** used?

- **Parallelizing legacy code**: If there are **global variables** in the existing code that need to be managed separately for each thread, `threadprivate` is useful.
- **Global variables in Fortran and C**: Commonly encountered in Fortran's _common blocks_ and C's _file-scope variables_.
- **Functions with stateful variables, such as random number generators**: If a function contains variables that must be kept separate for each thread, `threadprivate` ensures thread-local storage.

### Usage:

To make a variable _thread private_, use the `#pragma omp threadprivate(var)` directive.
```cpp
static int counter = 0;
#pragma omp threadprivate(counter) // This ensures that each thread has its **own independent copy** of the `counter` variable.
```

