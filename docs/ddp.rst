Distributed Training
====================

The library provides a DDP wrapper for distributed training with gradient synchronization
and eval cache coordination.

Basic usage
-----------

.. code-block:: python

   from tensor_ring_decomposition import TensorRingEmbedding, TensorRingDDP
   from torch.nn.parallel import DistributedDataParallel

   emb = TensorRingEmbedding(50000, 768, rank=8)
   ddp_emb = TensorRingDDP(emb)
   ddp_emb = DistributedDataParallel(ddp_emb, device_ids=[local_rank])

   # Training loop
   for batch in dataloader:
       output = ddp_emb(batch["input_ids"])
       loss = output.sum()
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()

   # Eval mode with cross-rank cache sync
   ddp_emb.to_eval_mode()

Manual gradient sync
--------------------

If not using PyTorch's ``DistributedDataParallel`` directly:

.. code-block:: python

   ddp_emb = TensorRingDDP(emb)
   loss.backward()
   ddp_emb.sync_gradients()  # all-reduce across ranks
   optimizer.step()
