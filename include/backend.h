/**
 * @file backend.h
 * @brief Unified training backend interface for BPTT learning.
 *
 * Abstraction layer that unifies CPU and OpenCL training paths under a
 * common API.  Main (bptt_learning.cpp) calls into the interface without
 * knowing which backend is active — the concrete implementation is chosen
 * at runtime via a factory function.
 *
 * Design goals
 * ------------
 * 1. Single training loop in main — no #ifdef or if/else branches per backend.
 * 2. Each backend owns its own internal state (GPU buffers, thread pools,
 *    Adam moments, etc.) and manages its own lifetime.
 * 3. The interface exposes only what main needs: epoch execution, metric
 *    queries, and weight synchronization back to the neuro::Network object.
 *
 * Typical usage
 * -------------
 * @code
 *   // 1. Create backend via factory
 *   std::unique_ptr<TrainingBackend> backend =
 *       create_backend(cfg, n, nc, train, test, state, lr, decay);
 *
 *   // 2. Training loop
 *   for (size_t epoch = 0; epoch < epochs; ++epoch) {
 *       backend->do_one_epoch(epoch);
 *       auto stats = backend->get_stats();
 *       printf("Epoch %zu — train: acc=%.4f loss=%.6f | "
 *              "test:  acc=%.4f loss=%.6f\n",
 *              epoch, stats.train_acc, stats.train_loss,
 *              stats.test_acc,  stats.test_loss);
 *   }
 *
 *   // 3. Sync trained weights back into the neuro::Network so it can be
 *   //    serialized / inspected / used for inference.
 *   backend->update_weights(n);
 * @endcode
 *
 * Implementer notes
 * -----------------
 * - `do_one_epoch()` MUST perform both training forward+backward passes AND
 *   a test-set forward pass so that get_stats() can return all four metrics.
 * - Batching, shuffling, and mini-batch SGD logic live INSIDE the backend;
 *   main only drives the epoch counter.
 * - `update_weights()` writes the current trained weights into the
 *   `neuro::Network` edge objects.  The caller is responsible for ensuring
 *   the Network's internal representation is in sync before serialization.
 * - Backends are non-copyable.  Use std::unique_ptr for ownership.
 */

#pragma once

#include "shared.h"
#include "training.h"
#include <cstddef>
#include <memory>
#include <utility>

/**
 * @struct TrainingStats
 * @brief Snapshot of training and test-set metrics for a single epoch.
 *
 * Returned by `TrainingBackend::get_stats()` after each call to
 * `do_one_epoch()`.  Values are averaged over the full dataset
 * (not per-batch).
 *
 * Fields
 * ------
 * - train_acc  : Fraction of correctly classified training samples [0, 1].
 * - train_loss : Mean cross-entropy loss over the training set.
 * - test_acc   : Fraction of correctly classified test samples [0, 1].
 *               Set to 0.0 if no test set was provided.
 * - test_loss  : Mean cross-entropy loss over the test set.
 *               Set to 0.0 if no test set was provided.
 */
struct TrainingStats {
    double train_acc;   ///< Training accuracy (fraction correct, 0-1)
    double train_loss;  ///< Training loss (mean cross-entropy)
    double test_acc;    ///< Test accuracy (fraction correct, 0-1); 0.0 if no test data
    double test_loss;   ///< Test loss (mean cross-entropy); 0.0 if no test data
};

/**
 * @class TrainingBackend
 * @brief Pure-virtual interface for BPTT training backends.
 *
 * Concrete implementations:
 *   - CpuBackend     — multi-threaded CPU BPTT (forward_backward.cpp path)
 *   - OpenclBackend  — GPU-accelerated BPTT via OpenCL kernels
 *
 * The interface is designed so that main's training loop is backend-agnostic:
 * @code
 *   for (size_t e = 0; e < epochs; ++e) {
 *       backend->do_one_epoch(e);
 *       auto s = backend->get_stats();
 *       // print / log / checkpoint …
 *   }
 *   backend->update_weights(network);
 * @endcode
 *
 * Thread-safety
 * -------------
 * Implementations must be safe for a single caller (main thread).  Internal
 * parallelism (OpenMP, pthreads, GPU work-groups) is managed by the backend
 * itself and is not visible through this interface.
 */
class TrainingBackend {
public:
    virtual ~TrainingBackend() = default;

    /**
     * @brief Execute one complete training + evaluation epoch.
     *
     * Contract
     * --------
     * 1. Shuffle the training data order (for SGD).
     * 2. Iterate over all training samples in mini-batches:
     *    - Forward pass (encode input → simulate → collect outputs)
     *    - Loss computation (cross-entropy against labels)
     *    - Backward pass (BPTT gradient accumulation)
     *    - Weight update (Adam or equivalent optimizer)
     * 3. After all training batches, run a full forward pass over the
     *    test set (no gradients, no weight updates) to compute test metrics.
     * 4. Accumulate and average metrics so get_stats() returns epoch-level
     *    values (not batch-level).
     *
     * Implementer details
     * -------------------
     * - The @p epoch parameter is 0-based.  Implementations may use it for
     *   learning-rate scheduling, logging, or checkpoint decisions.
     * - Adam bias-correction terms (b1_t, b2_t) must be updated internally
     *   across calls; do NOT reset them each epoch.
     * - For the OpenCL backend: GPU kernel timing / profiling should be
     *   collected here if timing is enabled.
     *
     * @param epoch  Current epoch index (0-based).
     */
    virtual void do_one_epoch(size_t epoch) = 0;

    /**
     * @brief Return the averaged metrics for the most recent epoch.
     *
     * Must be called AFTER `do_one_epoch()` returns.  Returns a copy of the
     * current epoch's training and test metrics.
     *
     * @return TrainingStats struct with train_acc, train_loss, test_acc, test_loss.
     *
     * Note: If `do_one_epoch()` has not yet been called, return values are
     * undefined (implementations should initialize to 0 or NaN).
     */
    virtual TrainingStats get_stats() const = 0;

    /**
     * @brief Synchronize trained weights back into the neuro::Network object.
     *
     * During training, weights live in a backend-optimized representation:
     *   - CPU backend:  std::vector<std::vector<double>> in TrainingState
     *   - OpenCL backend: GPU buffer of packed short/int values
     *
     * This method reads the current trained weights from the backend's
     * internal representation and writes them into the corresponding
     * `neuro::Edge` objects inside the Network, so the Network is ready
     * for:
     *   - JSON serialization (n->to_json)
     *   - CPU-only inference / evaluation
     *   - Checkpoint / resume
     *
     * Implementer details
     * -------------------
     * - For OpenCL: must read GPU weight buffer back to host memory first,
     *   then apply scale_factor to convert from discrete short units to
     *   floating-point weights.
     * - For discrete networks, apply the inverse of the quantization scaling
     *   (divide by scale_factor) when writing to Network edges.
     * - For non-discrete networks, write weights directly (scale_factor = 1.0).
     * - Only iterate over existing edges (n->get_node(i)->incoming); do NOT
     *   assume a dense weight matrix.
     *
     * @param network  Pointer to the neuro::Network to update.  Must not be null.
     */
    virtual void update_weights(neuro::Network* network) = 0;

    /**
     * @brief Run a final CPU forward pass on the test set (OpenCL only).
     *
     * For OpenCL backend: reads GPU weights to host, writes to network edges,
     * runs CPU forward+loss on test set, prints "Final CPU Test Loss/Acc".
     * For CPU backend: no-op, returns {0, 0}.
     *
     * @return {test_loss, test_accuracy} from final CPU evaluation.
     */
    virtual std::pair<double, double> run_final_cpu_eval() const { return {0.0, 0.0}; }

    /**
     * @brief Finalize training: export JSON, print summary.
     *
     * Called once after all epochs complete. For OpenCL backend: reads
     * GPU weights, syncs to network, runs final CPU eval, exports JSON.
     * For CPU backend: exports JSON only (weights already synced).
     */
    virtual void finalize() { }
};

/**
 * @brief Factory function to create the appropriate backend for the given config.
 *
 * Inspects `cfg.opencl` to decide which backend to instantiate.  All backend
 * initialization (device selection, kernel creation, thread pool setup, buffer
 * allocation, data encoding) happens inside the factory — main does not need
 * to know backend-specific setup steps.
 *
 * @param cfg          Parsed CLI configuration (contains opencl flag, timings, etc.)
 * @param n            Pointer to the neuro::Network (shared ownership; backend
 *                     reads topology but main retains ownership for cleanup).
 * @param nc           Network configuration (neuron counts, timesteps, thresholds…).
 * @param train        Training dataset.
 * @param test         Test dataset (may be empty).
 * @param state        Pre-allocated TrainingState holding initial weights,
 *                     delays, thresholds, and thread args.  Backend may take
 *                     ownership of internal buffers.
 * @param batch_size   Mini-batch size for SGD.
 * @param learning_rate  Adam learning rate (alpha).
 * @param decay_rate     L2 weight decay coefficient.
 *
 * @return unique_ptr to the concrete TrainingBackend implementation.
 *
 * @throw std::runtime_error if OpenCL is requested but no suitable device
 *        is available, or if OpenCL is requested for a non-discrete network.
 */
std::unique_ptr<TrainingBackend> create_backend(
    const CliConfig& cfg,
    neuro::Network* n,
    NetworkConfiguration& nc,
    const Dataset& train,
    const Dataset& test,
    TrainingState* state,
    size_t batch_size,
    double learning_rate,
    double decay_rate,
    double rho,
    double tau);

