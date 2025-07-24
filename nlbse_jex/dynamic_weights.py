##########################################################
# original code from https://github.com/Cranial-XIX/FAMO #
##########################################################

import torch
import torch.nn.functional as F
from typing import Union


class FAMO:
    """
    Fast Adaptive Multitask Optimization.
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 0.01,   # the regularization coefficient
        w_lr: float = 0.025,   # the learning rate of the task logits
        max_norm: float = 1.0, # the maximum gradient norm
    ):
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
        self,
        losses: torch.Tensor,
    ) -> Union[torch.Tensor, None]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        Returns
        -------
        Loss, extra outputs
        """
        loss = self.get_weighted_loss(losses=losses)
        loss.backward()
        return loss


##########################################################
# coda adapted from https://arxiv.org/pdf/2410.06741     #
##########################################################

# TODO adapt to out case. now work with MoE and specific designed collate_fn and DataLoader.

# how to use is https://github.com/codefuse-ai/MFTCoder/blob/main/mftcoder_accelerate/src/pefts/mft_trainer.py#L423
# # Training Loop!
#         for epoch in range(starting_epoch, self.args.num_train_epochs):
#             # set_epoch
#             # self.train_dataloader.set_epoch(epoch)

#             # if we early stop by some ckpts not converging
#             if self.args.early_stopping and stall_num == self.args.early_stopping_stall_num:
#                 break

#             if self.args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
#                 # We skip the first `n` batches in the dataloader when resuming from a checkpoint
#                 active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step)
#             else:
#                 active_dataloader = self.train_dataloader
#             tail_num = len(active_dataloader) - len(active_dataloader) % self.args.gradient_accumulation_steps
#             print(f"length of dataloader: {len(active_dataloader)}")

#             self.model.train()
#             # Inner Loop!
#             for step, batch in enumerate(active_dataloader):
#                 if step == tail_num:
#                     break
#                 with self.accelerator.accumulate(self.model):
#                     if step == 0:
#                         self.touch(batch, num_tokens=10)
#                     # forward
#                     outputs = self.model(
#                         input_ids=batch["input_ids"],
#                         attention_mask=batch["attention_mask"],
#                         position_ids=batch["position_ids"],
#                         return_dict=True,
#                     )

#                     if (
#                         self.args.weighted_loss_mode == "coba"
#                         and self.accelerator.sync_gradients
#                         and completed_steps % self.args.coba_update_interval == 0
#                         and completed_steps >= self.args.coba_warmup_steps
#                     ):
#                         with torch.no_grad():
#                             per_task_weight = coba_status.compute_per_task_weight(completed_steps=completed_steps)
#                             coba_status.log_per_task_weight += per_task_weight
#                             # logger.info(f'per_task_weight: {per_task_weight}', main_process_only=True)

#                     # loss
#                     loss, task_loss, _ = loss_func_mft(
#                         outputs=outputs,
#                         labels=batch["labels"],
#                         task_mask=batch["task_mask"],
#                         task_id=batch["task_id"],
#                         weighted_loss_mode=self.args.weighted_loss_mode,
#                         loss_mask=batch["loss_mask"],
#                         task_weights=per_task_weight,
#                     )

#                     # accelerator.print(len(outputs.router_logits), outputs.router_logits[0], outputs.router_logits[-1])
#                     # accelerator.print(batch['attention_mask'].shape, batch['attention_mask'])
#                     aux_loss = None
#                     if hasattr(self.model_config, "output_router_logits") and self.model_config.output_router_logits:
#                         if hasattr(self.model_config, "num_local_experts"):
#                             num_experts = self.model_config.num_local_experts
#                         elif hasattr(self.model_config, "num_experts"):
#                             num_experts = self.model_config.num_experts
#                         else:
#                             raise ValueError("model has no attribute num_local_experts or num_experts")
#                         aux_loss = load_balancing_loss_func(
#                             outputs.router_logits,
#                             num_experts,
#                             self.model_config.num_experts_per_tok,
#                             batch["attention_mask"],
#                         )
#                         aux_loss = self.model_config.router_aux_loss_coef * aux_loss.to(loss.device)
#                         loss += aux_loss  # make sure to reside in the same device

#                     # backward
#                     self.accelerator.backward(loss)
#                     # print(self.lr_scheduler.state_dict(), self.accelerator.process_index)
#                     # update(sync_gradients)
#                     self.optimizer.step()
#                     self.lr_scheduler.step()
#                     self.optimizer.zero_grad()
#                     # support args.min_lr
#                     if self.optimizer.param_groups[0]["lr"] <= self.args.min_lr:
#                         self.optimizer.param_groups[0]["lr"] = self.args.min_lr

#                     # accumulate resuce_loss and reduce_task_loss in a log_interval
#                     if not torch.isnan(loss):
#                         reduce_loss += loss.detach().float()
#                     if aux_loss and not torch.isnan(aux_loss):
#                         reduce_aux_loss += aux_loss.detach().float()
#                     # self.print("task loss devices: ", reduce_task_loss.device, task_loss.device)
#                     reduce_task_loss += task_loss.detach().float()
#                     reduce_task_exist += (task_loss != 0).detach().float()

#                     # If the accelerator has performed an optimization step behind the scenes, thus a completed_step done.
#                     if self.accelerator.sync_gradients:
#                         if (
#                             self.args.weighted_loss_mode == "coba"
#                             and completed_steps % self.args.coba_update_interval == 0
#                             and completed_steps >= 1
#                         ):
#                             coba_status.sample_valid_batch(self.model, completed_steps)
#                             # logger.info(f"valid_task_loss: {coba_status.valid_task_loss_accumulated}", main_process_only=True)

#                         # progress_bar.update(1)
#                         completed_steps += 1
#                         # monitoring training process and logging and tensorboarding
#                         if completed_steps % self.args.log_interval == 0:
#                             progress_bar.update(self.args.log_interval)
#                             if reduce_aux_loss > 0.0:
#                                 self.print(f"[INFO] aux_loss: {reduce_aux_loss/self.args.log_interval}")
#                             self.accelerate_monitor(
#                                 reduce_loss,
#                                 reduce_task_loss,
#                                 reduce_task_exist,
#                                 completed_steps,
#                                 coba_status,
#                             )
#                             # reset reduce_loss
#                             reduce_loss = torch.tensor(0.0).to(self.model.device)
#                             reduce_aux_loss = torch.tensor(0.0).to(self.model.device)
#                             reduce_task_loss = torch.zeros(len(ID2TASK)).to(self.model.device)
#                             reduce_task_exist = torch.zeros(len(ID2TASK)).to(self.model.device)

#                         # steps checkpointing
#                         if self.args.checkpointing_steps and completed_steps % self.args.checkpointing_steps == 0:
#                             output_dir = f"step_{completed_steps}"
#                             if self.args.output_dir is not None:
#                                 output_dir = os.path.join(self.args.output_dir, output_dir)
#                             self.accelerate_saving_checkpoint(output_dir, completed_steps)

#                         # steps evaluation
#                         if completed_steps % self.args.evaluation_steps == 0 and self.valid_dataloader:
#                             self.model.eval()
#                             eval_loss, eval_task_loss, min_eval_loss, stall_num, best_step = self.accelerate_evaluate(
#                                 completed_steps,
#                                 step,
#                                 min_eval_loss,
#                                 stall_num,
#                                 best_step,
#                             )
#                             self.model.train()

#                             # delete ckpts over args.saving_limit
#                             if self.accelerator.is_main_process and self.args.saving_limit:
#                                 delete_ckpts_over_limits(self.args.output_dir, self.args.saving_limit, best_step)

#                             # early stoppin when stalling more than args.early_stopping_stall_num
#                             if self.args.early_stopping and stall_num == self.args.early_stopping_stall_num:
#                                 self.print(f"[WARNING] Early stopping at {completed_steps}")
#                                 break

#                         if completed_steps >= self.args.max_train_steps:
#                             break
#                         self.accelerator.wait_for_everyone()



class MFTLossStatus:
    def __init__(self):
        super(MFTLossStatus, self).__init__()


class CoBaStatus(MFTLossStatus):
    def __init__(
        self,
        coba_warmup_steps=100,
        coba_history_length=200,
        coba_tau=5,
        coba_update_interval=1,
        coba_sample_valid_num=1,
        valid_dataloader=None,
        n_task=2
    ):

        super(CoBaStatus, self).__init__()
        self.coba_warmup_steps = coba_warmup_steps
        self.coba_history_length = coba_history_length
        self.coba_tau = coba_tau
        self.coba_update_interval = coba_update_interval
        self.coba_sample_valid_num = coba_sample_valid_num
        self.valid_dataloader = valid_dataloader
        self.valid_dataloader_length = len(valid_dataloader)
        self.valid_iterator = iter(valid_dataloader)
        self.valid_task_loss_accumulated = torch.zeros(n_task)
        self.history_task_valid_loss = None
        self.per_task_slope_list = None
        self.total_slope_list = None
        self.minimum_weight = 1 / (n_task * 10)
        self.valid_task_loss_begining = torch.ones(n_task, dtype=torch.float64)
        self.log_per_task_weight = torch.zeros(n_task)
        self.n_task = self.n_task

    def coba_evaluate(self, model, v_batch, per_task_weight=None, coba_status=None):
        model.eval()
        with torch.no_grad():
            valid_outputs = model(
                input_ids=v_batch["input_ids"],
                attention_mask=v_batch["attention_mask"],
                position_ids=v_batch["position_ids"],
            )

            _, valid_task_loss, valid_task_num = loss_func_mft(
                outputs=valid_outputs,
                labels=v_batch["labels"],
                task_mask=v_batch["task_mask"],
                task_id=v_batch["task_id"],
                weighted_loss_mode="coba",
                loss_mask=v_batch["loss_mask"],
                task_weights=None,
            )

            task_exist = (valid_task_loss != 0.0).float()
            torch.distributed.all_reduce(valid_task_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(task_exist, op=torch.distributed.ReduceOp.SUM)
            valid_task_loss /= task_exist.clamp_(1.0)
            valid_task_loss /= self.valid_task_loss_begining
        model.train()
        return valid_task_loss

    def compute_per_task_weight(self, completed_steps=None):
        task_num = self.n_task
        task_slope_fitting = torch.ones(task_num, dtype=torch.float64)
        start_step = max(0, completed_steps // self.coba_update_interval - self.coba_history_length)
        history_steps = torch.arange(start_step, completed_steps, 1)
        for i in range(task_num):
            per_task_history_valid_loss = self.history_task_valid_loss[i][-len(history_steps):]
            task_slope_fitting[i] = self.fit_window_slope(
                history_steps, per_task_history_valid_loss, type="slope"
            )
        history_total_valid_loss, index = torch.max(self.history_task_valid_loss[:, -len(history_steps):], dim=0)
        total_slope_fitting = self.fit_window_slope(
            history_steps, history_total_valid_loss, type="slope"
        )
        if completed_steps == self.coba_warmup_steps:
            self.per_task_slope_list = task_slope_fitting.unsqueeze(1)
            self.total_slope_list = total_slope_fitting.unsqueeze(0)
        else:
            self.per_task_slope_list = torch.cat((self.per_task_slope_list, task_slope_fitting.unsqueeze(1)), dim=-1)
            self.total_slope_list =  torch.cat((self.total_slope_list, total_slope_fitting.unsqueeze(0)), dim=0)
        
        # Relative Convergence Score
        normalize_task_slope = task_num * task_slope_fitting / task_slope_fitting.abs().sum()
        rcs = F.softmax(normalize_task_slope, dim=-1)
        
        # Absolute Convergence Score
        history_per_task_slope_list = self.per_task_slope_list[:, start_step:]
        reverse_normailize_iter_slope = -len(history_per_task_slope_list[0]) * history_per_task_slope_list \
                                        / history_per_task_slope_list.abs().sum(dim=-1, keepdim=True)

        flatten_rn_iter_slope = reverse_normailize_iter_slope.T.reshape(-1)
        current_step_rn_slope = flatten_rn_iter_slope[-task_num:]
        acs = F.softmax(current_step_rn_slope, dim=-1)

        # Divergence Factor
        normalize_total_iter_slope = - len(self.total_slope_list) * self.total_slope_list \
                                     / self.total_slope_list.abs().sum()
        divergence_factor = F.softmax(normalize_total_iter_slope * self.coba_tau, dim=-1)[-1] \
                          * len(self.total_slope_list)

        weight_logits = divergence_factor * rcs + (1 - divergence_factor) * acs
        per_task_weight = F.softmax(weight_logits * task_num, dim=-1)

        if len((per_task_weight < self.minimum_weight).nonzero().squeeze(0)) > 0:
            per_task_weight = per_task_weight * (1 - self.minimum_weight * task_num)
            per_task_weight += self.minimum_weight

        return per_task_weight
    
    def fit_window_slope(self, x, y, type="slope"):

        y = y[y != 0]
        x = x[:len(y)]
        
        nonzero_index = torch.squeeze(torch.nonzero(y), dim=1)
        y = torch.index_select(y, 0, nonzero_index)
        x = torch.index_select(x, 0, nonzero_index)

        ws = torch.flip(1 ** torch.arange(len(y)), dims=[0])
        ws = ws.double()

        if len(y) >= 2:
            if type == "slope":
                X = torch.stack((x, torch.ones_like(x, dtype=torch.float64))).T
                X = X.double()
            else:
                X = torch.stack((x ** 2, x, torch.ones_like(x, dtype=torch.float64))).T

            # implementation for numpy
            # X_np = X.T @ (ws[:, None] * X)
            # Y_np = X.T @ (ws * y)
            # w = torch.from_numpy(np.linalg.solve(X_np.numpy(), Y_np.numpy()))

            # implementation for torch
            w = torch.linalg.solve(X.T @ (ws[:, None] * X), X.T @ (ws * y))

            result = w[0]
        else:
            result = 0.0

        return result

    def sample_valid_batch(self, model, completed_steps):
        self.valid_task_loss_accumulated = torch.zeros(self.n_task, dtype=torch.float64)
        for i in range(self.coba_sample_valid_num):
            if (
                self.coba_sample_valid_num * completed_steps // self.coba_update_interval + i
            ) % self.valid_dataloader_length == 0:
                self.valid_iterator = iter(self.valid_dataloader)
                v_batch = next(self.valid_iterator)
            else:
                v_batch = next(self.valid_iterator)
            valid_task_loss = self.coba_evaluate(model, v_batch)
            self.valid_task_loss_accumulated += valid_task_loss.detach().cpu().double()

        self.valid_task_loss_accumulated /= self.coba_sample_valid_num
        if self.history_task_valid_loss is None and completed_steps >= 1:
            self.history_task_valid_loss = self.valid_task_loss_accumulated.unsqueeze(1)
        elif self.history_task_valid_loss is not None:
            self.history_task_valid_loss = torch.cat(
                (self.history_task_valid_loss, self.valid_task_loss_accumulated.unsqueeze(1)), dim=-1
            )