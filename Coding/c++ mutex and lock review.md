# c++ mutex and lock review

# C++ Concurrency: Mutex, Unique_lock, and Condition_variable Tutorial

## Mutex (Mutual Exclusion)

A mutex is a synchronization primitive that prevents multiple threads from accessing a shared resource simultaneously. It acts like a lock that only one thread can hold at a time.

**Key points:**

- `std::mutex` provides basic locking functionality
- `lock()` acquires the mutex, `unlock()` releases it
- If a thread tries to lock an already locked mutex, it blocks until the mutex becomes available

## Unique_lock

`std::unique_lock` is a flexible wrapper around mutex that provides RAII (Resource Acquisition Is Initialization) for mutex management. It automatically unlocks the mutex when it goes out of scope.

**Advantages over raw mutex:**

- Automatic cleanup (RAII)
- Can be locked/unlocked multiple times
- Supports deferred locking, try-locking, and timed locking
- Movable (can transfer ownership)

## Condition_variable

`std::condition_variable` allows threads to wait for certain conditions to be met. It works with `std::unique_lock<std::mutex>` to provide efficient thread synchronization.

**Key methods:**

- `wait()` - blocks until notified
- `notify_one()` - wakes up one waiting thread
- `notify_all()` - wakes up all waiting threads## Key Concepts Demonstrated

**Mutex Usage:**

- Protects shared data from race conditions
- Only one thread can hold the lock at a time
- Used with RAII wrappers for automatic cleanup

**Unique_lock Benefits:**

- Automatic lock/unlock management
- Can be unlocked early and relocked
- Works with condition variables
- Movable and flexible

**Condition_variable Patterns:**

- **wait()** with predicates to avoid spurious wakeups
- **wait_for()** with timeouts
- **notify_one()** vs **notify_all()** for different scenarios

## Common Use Cases

**Producer-Consumer:** Multiple threads producing/consuming from a shared buffer

**Thread Synchronization:** Coordinating thread execution order

**Resource Management:** Limiting access to limited resources

**Event Signaling:** One thread waiting for events from another

## Best Practices

1. Always use RAII wrappers (`unique_lock`, `lock_guard`) instead of raw mutex operations
2. Use condition variables with predicates to handle spurious wakeups
3. Keep critical sections as small as possible
4. Consider using `std::lock_guard` for simple lock-and-hold scenarios
5. Use `std::unique_lock` when you need more flexibility
6. Always check conditions in a loop when using condition variables

The demo code shows a complete producer-consumer implementation with proper synchronization, demonstrating real-world usage of these concurrency primitives.

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <random>

class ProducerConsumerDemo {
private:
    std::queue<int> buffer;
    std::mutex mtx;                    // Protects shared data
    std::condition_variable cv_producer; // For producer synchronization
    std::condition_variable cv_consumer; // For consumer synchronization
    
    const size_t MAX_BUFFER_SIZE = 5;
    bool finished = false;
    
public:
    void producer(int id) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(100, 999);
        
        for (int i = 0; i < 10; ++i) {
            int item = dis(gen);
            
            // Use unique_lock for flexible locking
            std::unique_lock<std::mutex> lock(mtx);
            
            // Wait until buffer has space
            cv_producer.wait(lock, [this] { 
                return buffer.size() < MAX_BUFFER_SIZE; 
            });
            
            // Produce item
            buffer.push(item);
            std::cout << "Producer " << id << " produced: " << item 
                      << " (buffer size: " << buffer.size() << ")\n";
            
            // Notify consumers that new item is available
            cv_consumer.notify_one();
            
            // Simulate production time
            lock.unlock(); // Release lock while sleeping
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Signal completion
        std::unique_lock<std::mutex> lock(mtx);
        finished = true;
        cv_consumer.notify_all();
    }
    
    void consumer(int id) {
        while (true) {
            std::unique_lock<std::mutex> lock(mtx);
            
            // Wait until buffer has items or production is finished
            cv_consumer.wait(lock, [this] { 
                return !buffer.empty() || finished; 
            });
            
            // If no items and production finished, exit
            if (buffer.empty() && finished) {
                break;
            }
            
            // Consume item
            int item = buffer.front();
            buffer.pop();
            std::cout << "Consumer " << id << " consumed: " << item 
                      << " (buffer size: " << buffer.size() << ")\n";
            
            // Notify producers that space is available
            cv_producer.notify_one();
            
            // Simulate consumption time
            lock.unlock(); // Release lock while sleeping
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        
        std::cout << "Consumer " << id << " finished\n";
    }
};

// Demonstration of different locking techniques
void lockingTechniquesDemo() {
    std::mutex mtx;
    int shared_data = 0;
    
    std::cout << "\n=== Locking Techniques Demo ===\n";
    
    auto worker = [&mtx, &shared_data](int id) {
        for (int i = 0; i < 5; ++i) {
            // Method 1: Basic unique_lock (RAII)
            {
                std::unique_lock<std::mutex> lock(mtx);
                int old_value = shared_data;
                shared_data += id;
                std::cout << "Thread " << id << ": " << old_value 
                          << " -> " << shared_data << std::endl;
            } // lock automatically released here
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };
    
    std::thread t1(worker, 1);
    std::thread t2(worker, 10);
    
    t1.join();
    t2.join();
    
    std::cout << "Final shared_data value: " << shared_data << std::endl;
}

void advancedLockingDemo() {
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    
    std::cout << "\n=== Advanced Locking Demo ===\n";
    
    // Thread that waits for signal
    std::thread waiter([&mtx, &cv, &ready] {
        std::unique_lock<std::mutex> lock(mtx);
        std::cout << "Waiter: Waiting for signal...\n";
        
        // Wait with timeout (wait for 2 seconds max)
        if (cv.wait_for(lock, std::chrono::seconds(2), [&ready] { return ready; })) {
            std::cout << "Waiter: Received signal!\n";
        } else {
            std::cout << "Waiter: Timeout occurred!\n";
        }
    });
    
    // Thread that sends signal
    std::thread notifier([&mtx, &cv, &ready] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        {
            std::lock_guard<std::mutex> lock(mtx); // Alternative to unique_lock for simple cases
            ready = true;
            std::cout << "Notifier: Sending signal\n";
        }
        cv.notify_one();
    });
    
    waiter.join();
    notifier.join();
}

// WARNING: This function demonstrates deadlock - DO NOT USE IN PRODUCTION
void deadlockDemo() {
    std::cout << "\n=== Deadlock Demo (BAD PRACTICE) ===\n";
    std::cout << "WARNING: This demonstrates what NOT to do!\n\n";
    
    std::mutex mutex1, mutex2;
    int resource1 = 0, resource2 = 0;
    bool enable_deadlock = false; // Set to true to see actual deadlock
    
    std::cout << "Choose demo type:\n";
    std::cout << "1. Safe version (deadlock avoided)\n";
    std::cout << "2. Deadlock version (commented out for safety)\n";
    
    if (!enable_deadlock) {
        std::cout << "\nRunning SAFE version with deadlock prevention...\n";
        
        // SAFE VERSION: Always acquire locks in the same order
        auto safeWorker = [&](int id, int iterations) {
            for (int i = 0; i < iterations; ++i) {
                // Always lock mutex1 first, then mutex2 (consistent order)
                std::lock_guard<std::mutex> lock1(mutex1);
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Simulate work
                std::lock_guard<std::mutex> lock2(mutex2);
                
                resource1 += id;
                resource2 += id * 2;
                std::cout << "Thread " << id << ": resource1=" << resource1 
                          << ", resource2=" << resource2 << std::endl;
                
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };
        
        std::thread t1(safeWorker, 1, 3);
        std::thread t2(safeWorker, 10, 3);
        
        t1.join();
        t2.join();
        
        std::cout << "Safe version completed successfully!\n";
    }
    
    std::cout << "\n--- Deadlock Prevention Techniques ---\n";
    
    // Technique 1: std::lock() for multiple mutexes
    auto demonstrateLockFunction = [&]() {
        std::cout << "\nTechnique 1: Using std::lock() for multiple mutexes\n";
        
        auto worker = [&](int id) {
            for (int i = 0; i < 3; ++i) {
                // std::lock acquires multiple locks atomically
                std::unique_lock<std::mutex> lock1(mutex1, std::defer_lock);
                std::unique_lock<std::mutex> lock2(mutex2, std::defer_lock);
                std::lock(lock1, lock2); // Deadlock-free acquisition
                
                resource1 += id;
                resource2 -= id;
                std::cout << "Thread " << id << " (std::lock): resource1=" 
                          << resource1 << ", resource2=" << resource2 << std::endl;
                
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        };
        
        std::thread t1(worker, 2);
        std::thread t2(worker, 20);
        t1.join();
        t2.join();
    };
    
    demonstrateLockFunction();
    
    // Technique 2: scoped_lock (C++17)
    auto demonstrateScopedLock = [&]() {
        std::cout << "\nTechnique 2: Using std::scoped_lock (C++17)\n";
        
        auto worker = [&](int id) {
            for (int i = 0; i < 2; ++i) {
                // scoped_lock automatically handles multiple mutexes safely
                std::scoped_lock lock(mutex1, mutex2);
                
                resource1 *= 2;
                resource2 += id * 3;
                std::cout << "Thread " << id << " (scoped_lock): resource1=" 
                          << resource1 << ", resource2=" << resource2 << std::endl;
                
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        };
        
        std::thread t1(worker, 1);
        std::thread t2(worker, 5);
        t1.join();
        t2.join();
    };
    
    demonstrateScopedLock();
    
    // Technique 3: try_lock with timeout
    auto demonstrateTryLock = [&]() {
        std::cout << "\nTechnique 3: Using try_lock to avoid indefinite waiting\n";
        
        auto worker = [&](int id) {
            for (int i = 0; i < 3; ++i) {
                std::unique_lock<std::mutex> lock1(mutex1, std::try_to_lock);
                if (lock1.owns_lock()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    
                    std::unique_lock<std::mutex> lock2(mutex2, std::try_to_lock);
                    if (lock2.owns_lock()) {
                        resource1 += id;
                        resource2 -= id;
                        std::cout << "Thread " << id << " (try_lock): Both locks acquired, "
                                  << "resource1=" << resource1 << ", resource2=" << resource2 << std::endl;
                    } else {
                        std::cout << "Thread " << id << ": Could not acquire mutex2, skipping\n";
                    }
                } else {
                    std::cout << "Thread " << id << ": Could not acquire mutex1, skipping\n";
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        };
        
        std::thread t1(worker, 3);
        std::thread t2(worker, 7);
        t1.join();
        t2.join();
    };
    
    demonstrateTryLock();
}

/*
DEADLOCK EXAMPLE (COMMENTED OUT FOR SAFETY):

void actualDeadlockExample() {
    // DO NOT UNCOMMENT - THIS WILL HANG YOUR PROGRAM!
    
    std::mutex mutex1, mutex2;
    
    // Thread 1: locks mutex1 then mutex2
    std::thread t1([&]() {
        std::lock_guard<std::mutex> lock1(mutex1);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::lock_guard<std::mutex> lock2(mutex2);  // Will deadlock here
        // ... work ...
    });
    
    // Thread 2: locks mutex2 then mutex1 (OPPOSITE ORDER)
    std::thread t2([&]() {
        std::lock_guard<std::mutex> lock2(mutex2);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::lock_guard<std::mutex> lock1(mutex1);  // Will deadlock here
        // ... work ...
    });
    
    t1.join();  // Will never return
    t2.join();  // Will never return
}
*/

int main() {
    std::cout << "=== Producer-Consumer Demo ===\n";
    
    ProducerConsumerDemo demo;
    
    // Create producer and consumer threads
    std::thread producer_thread(&ProducerConsumerDemo::producer, &demo, 1);
    std::thread consumer1_thread(&ProducerConsumerDemo::consumer, &demo, 1);
    std::thread consumer2_thread(&ProducerConsumerDemo::consumer, &demo, 2);
    
    // Wait for all threads to complete
    producer_thread.join();
    consumer1_thread.join();
    consumer2_thread.join();
    
    // Run additional demos
    lockingTechniquesDemo();
    advancedLockingDemo();
    
    return 0;
}
```