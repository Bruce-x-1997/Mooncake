#!/usr/bin/env python3
"""
Test script for mooncake-pg send/recv interfaces.

This script demonstrates how to use mooncake-pg's send and recv functions
for point-to-point communication between distributed processes.

Usage:
    # Run with 2 processes
    python -m torch.distributed.launch --nproc_per_node=2 test_pg_send_recv.py

    # Or use mp.spawn
    python test_pg_send_recv.py --distributed
"""

import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np


# Removed setup_distributed and cleanup_distributed - now using inline initialization


def sync_processes(use_cuda=False):
    """Synchronize processes. Uses barrier for CPU, all_reduce for CUDA."""
    if use_cuda:
        # CUDA mode: use all_reduce for synchronization (barrier not supported)
        sync_tensor = torch.tensor([1], dtype=torch.int32, device="cuda")
        dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    else:
        # CPU mode: use barrier
        dist.barrier()


def print_latency_stats(operation_name, latencies_ms, rank=0):
    """Print latency statistics."""
    if not latencies_ms:
        return
    
    latencies_ms = np.array(latencies_ms)
    mean_lat = np.mean(latencies_ms)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)
    median_lat = np.median(latencies_ms)
    p50 = np.percentile(latencies_ms, 50)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)
    
    print(f"\n[Rank {rank}] {operation_name} Latency Statistics:")
    print(f"  Count: {len(latencies_ms)}")
    print(f"  Mean:  {mean_lat:.3f} ms")
    print(f"  Min:   {min_lat:.3f} ms")
    print(f"  Max:   {max_lat:.3f} ms")
    print(f"  Median: {median_lat:.3f} ms")
    print(f"  P50:   {p50:.3f} ms")
    print(f"  P95:   {p95:.3f} ms")
    print(f"  P99:   {p99:.3f} ms")


def test_basic_send_recv(rank, world_size, use_cuda=True, use_pytorch_api=False):
    """Test basic send/recv functionality.
    
    Args:
        use_pytorch_api: If True, use PyTorch's P2POp API (dist.isend/irecv).
                        If False, use backend's send/recv directly.
    """
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Test: Basic Send/Recv (API: {'PyTorch' if use_pytorch_api else 'Backend'})")
    print(f"{'='*60}")
    
    backend_name = "mooncake" if use_cuda else "mooncake-cpu"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    from mooncake import pg
    
    dist.init_process_group(
        backend=backend_name,
        rank=rank,
        world_size=world_size,
        pg_options=pg.MooncakeBackendOptions(
            torch.zeros((world_size,), dtype=torch.int32, 
                       device="cuda" if use_cuda else "cpu")
        ),
    )
    
    try:
        # Create test tensor
        tensor_size = (4, 4)
        if use_cuda:
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
        
        # Use float32 to ensure consistent dtype
        send_tensor = torch.ones(tensor_size, device=device, dtype=torch.float32) * (rank + 1)
        recv_tensor = torch.zeros(tensor_size, device=device, dtype=torch.float32)
        
        print(f"[Rank {rank}] Send tensor shape: {send_tensor.shape}, device: {send_tensor.device}")
        print(f"[Rank {rank}] Send tensor values:\n{send_tensor}")
        
        # Synchronize before communication
        sync_processes(use_cuda)
        
        latencies_ms = []
        
        if use_pytorch_api:
            # Method 1: Use PyTorch's P2POp API (recommended)
            if rank == 0:
                print(f"[Rank {rank}] Sending tensor to rank 1 using PyTorch API...")
                if use_cuda:
                    torch.cuda.synchronize()
                start_time = time.time()
                work = dist.isend(send_tensor, dst=1, tag=0)
                work.wait()
                if use_cuda:
                    torch.cuda.synchronize()
                elapsed_ms = (time.time() - start_time) * 1000
                latencies_ms.append(elapsed_ms)
                print(f"[Rank {rank}] Send completed in {elapsed_ms:.3f} ms")
            elif rank == 1:
                print(f"[Rank {rank}] Receiving tensor from rank 0 using PyTorch API...")
                if use_cuda:
                    torch.cuda.synchronize()
                start_time = time.time()
                work = dist.irecv(recv_tensor, src=0, tag=0)
                work.wait()
                if use_cuda:
                    torch.cuda.synchronize()
                elapsed_ms = (time.time() - start_time) * 1000
                latencies_ms.append(elapsed_ms)
                print(f"[Rank {rank}] Receive completed in {elapsed_ms:.3f} ms")
                print(f"[Rank {rank}] Received tensor values:\n{recv_tensor}")
        else:
            # Method 2: Use backend's send/recv directly (low-level)
            group = dist.group.WORLD
            backend = group._get_backend(device)
            print(f"[Rank {rank}] Backend: {type(backend).__name__}")
            
            if rank == 0:
                print(f"[Rank {rank}] Sending tensor to rank 1 using backend API...")
                if use_cuda:
                    torch.cuda.synchronize()
                start_time = time.time()
                work = backend.send([send_tensor], dstRank=1, tag=0)
                work.wait()
                if use_cuda:
                    torch.cuda.synchronize()
                elapsed_ms = (time.time() - start_time) * 1000
                latencies_ms.append(elapsed_ms)
                print(f"[Rank {rank}] Send completed in {elapsed_ms:.3f} ms")
            elif rank == 1:
                print(f"[Rank {rank}] Receiving tensor from rank 0 using backend API...")
                if use_cuda:
                    torch.cuda.synchronize()
                start_time = time.time()
                work = backend.recv([recv_tensor], srcRank=0, tag=0)
                work.wait()
                if use_cuda:
                    torch.cuda.synchronize()
                elapsed_ms = (time.time() - start_time) * 1000
                latencies_ms.append(elapsed_ms)
                print(f"[Rank {rank}] Receive completed in {elapsed_ms:.3f} ms")
                print(f"[Rank {rank}] Received tensor values:\n{recv_tensor}")
        
        if latencies_ms:
            print_latency_stats("Send/Recv", latencies_ms, rank)
        
        if rank == 1:
            # Verify correctness
            expected = torch.ones(tensor_size, device=device, dtype=torch.float32)
            if torch.allclose(recv_tensor, expected):
                print(f"[Rank {rank}] ✓ Verification passed: received correct values")
            else:
                print(f"[Rank {rank}] ✗ Verification failed: expected {expected}, got {recv_tensor}")
        
        sync_processes(use_cuda)
        print(f"[Rank {rank}] Test completed")
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        dist.destroy_process_group()


def test_bidirectional_send_recv(rank, world_size, use_cuda=True, use_pytorch_api=False):
    """Test bidirectional send/recv (both ranks send and receive)."""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Test: Bidirectional Send/Recv (API: {'PyTorch' if use_pytorch_api else 'Backend'})")
    print(f"{'='*60}")
    
    backend_name = "mooncake" if use_cuda else "mooncake-cpu"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    
    from mooncake import pg
    
    dist.init_process_group(
        backend=backend_name,
        rank=rank,
        world_size=world_size,
        pg_options=pg.MooncakeBackendOptions(
            torch.zeros((world_size,), dtype=torch.int32,
                       device="cuda" if use_cuda else "cpu")
        ),
    )
    
    try:
        # Create test tensors
        tensor_size = (8, 8)
        if use_cuda:
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
        
        # Use float32 to ensure consistent dtype between send and recv tensors
        # torch.arange() defaults to int64, while torch.zeros() defaults to float32
        send_tensor = torch.arange(64, device=device, dtype=torch.float32).reshape(8, 8) * (rank + 1)
        recv_tensor = torch.zeros(tensor_size, device=device, dtype=torch.float32)
        
        print(f"[Rank {rank}] Send tensor (first 4x4):\n{send_tensor[:4, :4]}")
        
        sync_processes(use_cuda)
        
        # Both ranks send and receive simultaneously
        other_rank = 1 - rank
        
        print(f"[Rank {rank}] Starting send to rank {other_rank} and recv from rank {other_rank}...")
        
        latencies_ms = []
        
        if use_pytorch_api:
            # Use PyTorch's batch_isend_irecv for bidirectional communication
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.time()
            p2p_ops = [
                dist.P2POp(op=dist.isend, tensor=send_tensor, peer=other_rank),
                dist.P2POp(op=dist.irecv, tensor=recv_tensor, peer=other_rank),
            ]
            works = dist.batch_isend_irecv(p2p_ops)
            for w in works:
                w.wait()
            if use_cuda:
                torch.cuda.synchronize()
            elapsed_ms = (time.time() - start_time) * 1000
            latencies_ms.append(elapsed_ms)
        else:
            # Use backend's send/recv directly
            group = dist.group.WORLD
            backend = group._get_backend(device)
            
            # Post both operations
            # Note: For bidirectional communication, we need to ensure send operations
            # start before recv operations to avoid deadlock. The recv operation will
            # block waiting for the sender's request, so if both recv operations start
            # before their corresponding send operations, they will deadlock.
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.time()
            
            # Start send operation first to avoid deadlock
            send_work = backend.send([send_tensor], dstRank=other_rank, tag=rank)
            
            # Give send operation a moment to start sending the request
            # This is necessary because recv will block waiting for the request
            time.sleep(0.001)  # 1ms should be enough for the request to be sent
            
            # Now start recv operation
            recv_work = backend.recv([recv_tensor], srcRank=other_rank, tag=other_rank)
            
            # Wait for both to complete
            send_work.wait()
            recv_work.wait()
            if use_cuda:
                torch.cuda.synchronize()
            elapsed_ms = (time.time() - start_time) * 1000
            latencies_ms.append(elapsed_ms)
        
        print(f"[Rank {rank}] Both operations completed in {latencies_ms[0]:.3f} ms")
        print_latency_stats("Bidirectional Send/Recv", latencies_ms, rank)
        print(f"[Rank {rank}] Received tensor (first 4x4):\n{recv_tensor[:4, :4]}")
        
        # Verify correctness
        expected = torch.arange(64, device=device, dtype=torch.float32).reshape(8, 8) * (other_rank + 1)
        if torch.allclose(recv_tensor, expected):
            print(f"[Rank {rank}] ✓ Verification passed")
        else:
            print(f"[Rank {rank}] ✗ Verification failed")
            print(f"[Rank {rank}] Expected (first 4x4):\n{expected[:4, :4]}")
        
        sync_processes(use_cuda)
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        dist.destroy_process_group()


def run_test(rank, world_size, test_name, use_cuda, use_pytorch_api):
    """Run a specific test."""
    if test_name == "basic":
        test_basic_send_recv(rank, world_size, use_cuda, use_pytorch_api)
    elif test_name == "bidirectional":
        test_bidirectional_send_recv(rank, world_size, use_cuda, use_pytorch_api)
    elif test_name == "all":
        # Test both API styles
        test_basic_send_recv(rank, world_size, use_cuda, use_pytorch_api=False)
        test_basic_send_recv(rank, world_size, use_cuda, use_pytorch_api=True)
        test_bidirectional_send_recv(rank, world_size, use_cuda, use_pytorch_api=False)
        test_bidirectional_send_recv(rank, world_size, use_cuda, use_pytorch_api=True)
    else:
        print(f"Unknown test: {test_name}")
        return


def main():
    parser = argparse.ArgumentParser(description="Test mooncake-pg send/recv interfaces")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["basic", "bidirectional", "all"],
        help="Test to run",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of processes",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of CUDA",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use mp.spawn for distributed execution",
    )
    parser.add_argument(
        "--pytorch-api",
        action="store_true",
        help="Use PyTorch's P2POp API instead of backend's send/recv directly",
    )
    
    args = parser.parse_args()
    
    use_cuda = not args.cpu and torch.cuda.is_available()
    
    if args.distributed:
        # Use mp.spawn
        print(f"Running with mp.spawn (world_size={args.world_size}, use_pytorch_api={args.pytorch_api})")
        mp.spawn(
            run_test,
            args=(args.world_size, args.test, use_cuda, args.pytorch_api),
            nprocs=args.world_size,
            join=True,
        )
    else:
        # Single process mode (for testing without distributed)
        print("Single process mode - use --distributed for multi-process testing")
        print("Or use: python -m torch.distributed.launch --nproc_per_node=2 test_pg_send_recv.py")
        print("\nExample commands:")
        print("  # Test with backend API (direct send/recv)")
        print("  python test_pg_send_recv.py --test basic --distributed")
        print("\n  # Test with PyTorch API (P2POp)")
        print("  python test_pg_send_recv.py --test basic --distributed --pytorch-api")
        print("\n  # Test all scenarios")
        print("  python test_pg_send_recv.py --test all --distributed")


if __name__ == "__main__":
    main()

