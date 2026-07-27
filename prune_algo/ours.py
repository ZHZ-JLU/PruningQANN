import math
import time

import math
import torch
import torch.nn as nn
import transformers
import numpy as np
import os
import logging
# from SNN.spike_layer import snnLinear
# from phase.phase_layer import phaseSnnLinear
from quantize.int_linear import QuantLinear

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
logger = logging.getLogger(__name__)

class ALPS_prune_ours:

    def __init__(self, layer, nsamples, seqlen, device, lamb):
        self.layer = layer
        self.device = device

        self.nsamples = nsamples
        self.seqlen = seqlen
        self.equi_nsamples = self.nsamples*self.seqlen
        self.lamb = lamb

        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.XtX = torch.zeros((self.columns, self.columns), device=self.device).float()

        self.count = 0

        self.nsamples = 0
     

    def add_batch(self, inp, out, extra=None):
        if extra is not None:
            inp = extra
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, (nn.Linear, QuantLinear)) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            if len(inp.shape) == 4:
                _, _, _, hidden = inp.shape
                inp = inp.reshape(-1, hidden) 
                
            inp = inp.t()
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
            if len(out.shape) == 4:
                out = out.reshape((-1, out.shape[-1]))
        out = out.t()
        if isinstance(self.layer, nn.Conv2d):
            print(inp.shape)
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        inp = inp.float()
        out = out.float()
        
        
        self.XtX += inp.matmul(inp.t())
        self.nsamples += tmp

    
    def ALPS_admm(self, sp=0.5, nm_n = 0, nm_m = 0, rho=0.1, max_iter = 300, update_iter = 3, switch_iter = 30, inner_iter=3):
        
        # get dense weight
        W = self.layer.weight.data.clone()
        W = W.float()
        W = W.to('cuda:0')
        self.XtX = self.XtX.to('cuda:0')
        W_q = self.layer.weight_quantizer.quant_weight(W.half()).float()
        S = self.layer.weight_quantizer.scales.data.reshape(-1).float().to(self.device)
        
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
            
        # ridge term
        damp1 = self.lamb * torch.mean(torch.diag(self.XtX))
        damp1 = damp1.to(device=self.XtX.device, dtype=self.XtX.dtype)
        diag = torch.arange(self.XtX.shape[0], device=self.XtX.device)
        self.XtX[diag, diag] += damp1
        
        # normalization 
        # X_norm = torch.Tensor([1.0]).float().to(self.device)
        X_norm = torch.diag(self.XtX).sqrt() + 1e-8
        self.XtX = self.XtX / X_norm
        self.XtX = (self.XtX.T / X_norm).T    
        
        self.YtX = torch.zeros_like(W)
        self.YtX = torch.matmul(W * X_norm,self.XtX).to(self.device)
        

        admm_st = time.time()

        # initialization
        # XTX_inv = torch.zeros_like(self.XtX).float().to(self.device)
        B = (W_q.to(self.device) * X_norm.to(self.device)).t().clone()
        W = (W.to(self.device) * X_norm.to(self.device)).t().clone()
        B_orig = B.clone()
        V = torch.zeros_like(B)
        D = torch.zeros_like(B)
        D_suppp = torch.zeros_like(B)
        D_supp = torch.zeros_like(B)

        
        totp, num_cout = B.shape
        L, Q = torch.linalg.eigh(self.XtX.double())
        # XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(self.device)
        
        init_rho = False
        fix_supp = False
        D_fix = torch.zeros_like(D)
        
        Res0 = self.YtX.T
        Res0 = torch.sum(B_orig.to(self.device)*S.to(self.device) * Res0.to(self.device))
        Res0 = torch.sum(Res0)

        params = B.shape[0]*B.shape[1]
        k_spar = int(np.round((1-sp)*params))
    
        
        if nm_n == 0:
            D = B.clone()
            D_tmp = (X_norm.to(self.device) * (D.T.to(self.device) / X_norm.to(self.device)).round().clamp(self.layer.weight_quantizer.qmin, self.layer.weight_quantizer.qmax)).T.to(self.device)
            delta_D = D - D_tmp
            D = D.reshape(-1)
            D_tmp = D_tmp.reshape(-1)
            delta_D = delta_D.reshape(-1)
            _, loss_idx = torch.topk(-(D**2-delta_D**2),totp * num_cout - k_spar)
            D_tmp[loss_idx] = 0    
            D = D_tmp
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(totp, num_cout)
        else:
            new_dim = totp * num_cout / nm_m
            new_dim = int(new_dim)
            k_spar = totp * num_cout * nm_n/nm_m
            
            D = B.clone()
            D_tmp = (X_norm * (D.T / X_norm).round().clamp(self.layer.weight_quantizer.qmin, self.layer.weight_quantizer.qmax)).T
            delta_D = D - D_tmp
            D = D.reshape((new_dim, nm_m))
            D_tmp = D_tmp.reshape((new_dim, nm_m))
            delta_D = delta_D.reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-(D**2-delta_D**2),nm_m - nm_n, dim = 1)
            # D_tmp[loss_idx] = 0    
            D_tmp = D_tmp.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(self.device),dim=1,index=loss_idx)   
            D = D_tmp
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(totp, num_cout)
            # D = B.clone().t().reshape((new_dim, nm_m))
            # _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
            # D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(self.device),dim=1,index=loss_idx)   
            # D_suppp = (D == 0).to(torch.float)
            # D = D.reshape(num_cout, totp).t()
    
        D_init = D.clone()
        errorp = 1
        last_S = S.clone()
        supp_change = 1e10
        op_s = True
        S = torch.diag(B.T.to(self.device) @ self.XtX.to(self.device) @ W.to(self.device)) / torch.diag(B.T.to(self.device) @ self.XtX.to(self.device) @ B.to(self.device))
        self.layer.weight_quantizer.scales.data = S.reshape(S.shape[0], -1).half()
        # self.layer.weight_quantizer.w_quant.data = (B.t() / X_norm.to(self.device)).half()
        # return 
        for i_admm in range(max_iter):
            # print("******************************************")
            # if i_admm>30 or inner_iter==0: #supp_change / k_spar < 0.002:
            #     op_s = True

            if fix_supp:
                D = ((V + rho * B) / rho) * D_fix
            elif nm_n == 0:
                D = ((V + rho * B) / rho)
                D_tmp = (X_norm * (D.T / X_norm).round().clamp(self.layer.weight_quantizer.qmin, self.layer.weight_quantizer.qmax)).T
                delta_D = D - D_tmp
                D = D.reshape(-1)
                D_tmp = D_tmp.reshape(-1)
                delta_D = delta_D.reshape(-1)
                _, loss_idx = torch.topk(-(D**2-delta_D**2),totp * num_cout - k_spar)
                D_tmp[loss_idx] = 0    
                D = D_tmp
                D = D.reshape(totp, num_cout)   
            else:
                D = ((V + rho * B) / rho)
                D_tmp = (X_norm * (D.T / X_norm).round().clamp(self.layer.weight_quantizer.qmin, self.layer.weight_quantizer.qmax)).T
                delta_D = D - D_tmp
                D = D.reshape((new_dim, nm_m))
                D_tmp = D_tmp.reshape((new_dim, nm_m))
                delta_D = delta_D.reshape((new_dim, nm_m))
                _, loss_idx = torch.topk(-(D**2-delta_D**2), nm_m - nm_n, dim = 1)
                D_tmp = D_tmp.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(self.device),dim=1,index=loss_idx)   
                # D_tmp[loss_idx] = 0    
                D = D_tmp
                D = D.reshape(totp, num_cout)   
                # D = ((V + rho * B) / rho).t().reshape((new_dim, nm_m))
                # _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
                # D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to('cuda:0'),dim=1,index=loss_idx) 
                # D_supp = (D == 0).to(torch.float)  
                # D = D.reshape(num_cout, totp).t()  

            if op_s:
                B_tmp = Q.T.float() @ (self.YtX.T*S.reshape(-1, S.shape[0]).to(self.device)-V+rho*D)
                inv = 1 / ((S**2).reshape(S.shape[0],-1).to(self.device) * L.repeat(S.shape[0],1).float() + rho)
                B = Q.float() @ (inv.T * B_tmp)
            else:
                print('here!')
                for inner_i in range(inner_iter):
                    # lmda = (inner_i+1) / inner_iter
                    
                    B_tmp = Q.T.float() @ (self.YtX.T*S.reshape(-1, S.shape[0])-V+rho*D)
                    inv = 1 / ((S**2).reshape(S.shape[0],-1) * L.repeat(S.shape[0],1).float() + rho)
                    B = Q.float() @ (inv.T * B_tmp)
                    # B = (1-lmda) * B + lmda * (Q.float() @ (inv.T * B_tmp))
                    
                    S = torch.diag(B.T @ self.XtX @ W) / torch.diag(B.T @ self.XtX @ B)
                    # S = (1-lmda) * S + lmda * (torch.diag(B.T @ self.XtX @ W) / torch.diag(B.T @ self.XtX @ B))
                    # print(torch.max(torch.abs((last_S-S))))
                    # if torch.mean(torch.abs((last_S-S))) < 1e-6:
                    #     print(inner_i)
                    #     break
                    last_S = S.clone()
                    # print(S)

            V = V + rho * (B - D)
            
            if (i_admm+1) % update_iter == 0:

         
                if nm_n == 0:
                    D_supp = (D.reshape(-1) == 0).to(torch.float)
                else:
                    D_supp = (D == 0).reshape(totp, num_cout).to(torch.float)
                    D_suppp = D_suppp.reshape(totp, num_cout).to(torch.float)

                supp_change = torch.sum((D_supp-D_suppp)**2)
                print(supp_change)
                
                if not fix_supp:
                    if supp_change / k_spar > 0.1:
                        init_rho = True
                        rho *= 1.3
                    elif supp_change / k_spar > 0.005:
                        init_rho = True
                        rho *= 1.2
                    elif supp_change > 0.5:
                        if init_rho:
                            rho *= 1.1
                        else:
                            rho /= 5
                            B = B_orig.clone().to(self.device)
                            D = D_init.clone().to(self.device)
                            V = torch.zeros_like(B).to(self.device)     
                    else:
                        if init_rho:
                            break
                        else:
                            rho /= 5
                
                D_suppp = (D_supp).clone()
                if rho > 1e6:
                    rho = 1e6
               
                # XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(self.device)
                
                if nm_n == 0:
                    Btest = B.clone()
                    Btest_tmp = (X_norm * (Btest.T / X_norm).round().clamp(self.layer.weight_quantizer.qmin, self.layer.weight_quantizer.qmax)).T
                    delta_Btest = Btest - Btest_tmp
                    Btest = Btest.reshape(-1)
                    Btest_tmp = Btest_tmp.reshape(-1)
                    delta_Btest = delta_Btest.reshape(-1)
                    _, loss_idx = torch.topk(-(Btest**2-delta_Btest**2),totp * num_cout - k_spar)
                    Btest_tmp[loss_idx] = 0    
                    Btest = Btest_tmp
                    Btest = Btest.reshape(totp, num_cout)
                else:
                    Btest = B.clone()
                    Btest_tmp = (X_norm * (Btest.T / X_norm).round().clamp(self.layer.weight_quantizer.qmin, self.layer.weight_quantizer.qmax)).T
                    delta_Btest = Btest - Btest_tmp
                    Btest = Btest.reshape((new_dim, nm_m))
                    Btest_tmp = Btest_tmp.reshape((new_dim, nm_m))
                    delta_Btest = delta_Btest.reshape((new_dim, nm_m))
                    _, loss_idx = torch.topk(-(Btest**2-delta_Btest**2), nm_m-nm_n, dim = 1)
                    # Btest_tmp[loss_idx] = 0    
                    Btest_tmp = Btest_tmp.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(self.device),dim=1,index=loss_idx)   
                    Btest = Btest_tmp
                    Btest = Btest.reshape(totp, num_cout)
                    
                    # _, loss_idx = torch.topk(-Btest**2,nm_m - nm_n, dim = 1)
                    # Btest = Btest.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(self.device),dim=1,index=loss_idx)  
                    # Btest = Btest.reshape(num_cout, totp).t()
            
                Resc = torch.matmul(self.XtX.to(self.device),Btest*S) - self.YtX.T
                Resc = torch.diag(torch.matmul((Btest*S-B_orig.to(self.device)*S).t(), Resc))

        
                errorc = torch.sum(Resc)/Res0
                errorc = errorc.item()
                
                print("iter {}, error {} support change {}, rho {}".format(i_admm, errorc / errorp, supp_change / k_spar, rho))
                
                
                if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                    break
            
        if nm_n == 0:
            B_tmp = (X_norm * (B.T / X_norm).round().clamp(self.layer.weight_quantizer.qmin, self.layer.weight_quantizer.qmax)).T
            delta_B = B - B_tmp
            B = B.reshape(-1)
            B_tmp = B_tmp.reshape(-1)
            delta_B = delta_B.reshape(-1)
            _, loss_idx = torch.topk(-(B**2-delta_B**2),totp * num_cout - k_spar)
            B_tmp[loss_idx] = 0    
            B = B_tmp
            B = B.reshape(totp, num_cout)
        else:
            B_tmp = (X_norm * (B.T / X_norm).round().clamp(self.layer.weight_quantizer.qmin, self.layer.weight_quantizer.qmax)).T
            delta_B = B - B_tmp
            B = B.reshape((new_dim, nm_m))
            B_tmp = B_tmp.reshape((new_dim, nm_m))
            delta_B = delta_B.reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-(B**2-delta_B**2),nm_m - nm_n, dim = 1)
            B_tmp = B_tmp.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to('cuda:0'),dim=1,index=loss_idx)    
            B = B_tmp
            B = B.reshape(totp, num_cout)
            # B = B.reshape(totp, num_cout)
            # B = B.t().reshape((new_dim, nm_m))
            # _, loss_idx = torch.topk(-B**2,nm_m - nm_n, dim = 1)
            # B = B.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to('cuda:0'),dim=1,index=loss_idx)  
            # B = B.reshape(num_cout, totp).t()

        V = None
        D = None

        error = None
        Res = torch.matmul(self.XtX,B*S) - self.YtX.T
        Res = torch.diag(torch.matmul((B*S-B_orig*S).t(), Res))
        
        error = torch.sum(Res)/Res0
        error = error.item()

        #print("Before backsolve, error is {}".format(error))
        admm_time = time.time() - admm_st
        
        # PCG
        back_st = time.time()
        # B = self.cg_batch( (self.XtX).to(self.device), self.YtX.T, 
        #                (B != 0).to(torch.float), M_bmm=None, X0=B, rtol=1e-4, atol=0., maxiter=10, verbose=False)
            
        back_time = time.time() - back_st
        
        
        Res = torch.matmul(self.XtX,B*S) - self.YtX.T
        Res = torch.diag(torch.matmul((B*S-B_orig*S).t(), Res))
        
        error = torch.sum(Res)/Res0
        error = error.item()
        
        torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            self.layer.weight.data = (B.t() / X_norm.to(self.device)).t().reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            # self.layer.weight.data = (B.t() / X_norm.to(self.device)).reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            # S = torch.diag(B.T @ self.XtX @ W) / torch.diag(B.T @ self.XtX @ B)
            self.layer.weight_quantizer.scales.data = S.reshape(S.shape[0], -1).half()
            self.layer.weight_quantizer.w_quant.data = ((B.t() / X_norm.to(self.device)).round().clamp(self.layer.weight_quantizer.qmin, self.layer.weight_quantizer.qmax)).half()
            # self.layer.weight_quantizer.quant_weight((B.t() / X_norm.to(self.device)).half())

        logger.info("Number of iter is {}".format(i_admm))
        logger.info("Final Error is {}".format(error))
        logger.info("Time is admm: {} back:{}".format(admm_time, back_time))

        return error
    
    
    
    # A modified version of https://github.com/sbarratt/torch_cg
    
    def cg_batch(self, A, B, A_supp, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

        This function solves matrix linear systems of the form

            A X = B,  

        where A is a n x n positive definite matrix and B is a n x m matrix,
        and X is the n x m matrix representing the solution for the ith system.

        Args:
            A_bmm: A callable that performs a batch matrix multiply of A and a n x m matrix.
            B: A n x m matrix representing the right hand sides.
            M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a n x m matrix. (default=identity matrix)
            X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
            rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
            atol: (optional) Absolute tolerance for norm of residual. (default=0)
            maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
            verbose: (optional) Whether or not to print status messages. (default=False)
        """
        error_list = np.zeros((maxiter,))
        n, m = B.shape

        if M_bmm is None:
            M_bmm = lambda x: x
        if X0 is None:
            X0 = M_bmm(B)
        if maxiter is None:
            maxiter = 5 * n

        assert B.shape == (n, m)
        assert X0.shape == (n, m)
        assert rtol > 0 or atol > 0
        assert isinstance(maxiter, int)

        X_k = X0
    
        R_k = B - A @ X_k
        R_k = R_k * A_supp
    
        Z_k = M_bmm(R_k)

        P_k = torch.zeros_like(Z_k)

        P_k1 = P_k
        R_k1 = R_k
        R_k2 = R_k
        X_k1 = X0
        Z_k1 = Z_k
        Z_k2 = Z_k

        B_norm = torch.norm(B, dim=1)
        stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

        if verbose:
            print("%03s | %010s %06s" % ("it", "dist", "it/s"))

        optimal = False
        start = time.perf_counter()
        for k in range(1, maxiter + 1):
            start_iter = time.perf_counter()
            Z_k = M_bmm(R_k)

            if k == 1:
                P_k = Z_k
                R_k1 = R_k
                X_k1 = X_k
                Z_k1 = Z_k
            else:
                R_k2 = R_k1
                Z_k2 = Z_k1
                P_k1 = P_k
                R_k1 = R_k
                Z_k1 = Z_k
                X_k1 = X_k
                denominator = (R_k2 * Z_k2).sum(0)
                denominator[denominator == 0] = 1e-8
                beta = (R_k1 * Z_k1).sum(0) / denominator
                P_k = Z_k1 + beta.unsqueeze(0) * P_k1

            denominator = (P_k * (A@P_k)).sum(0)
            denominator[denominator == 0] = 1e-8
            alpha = (R_k1 * Z_k1).sum(0) / denominator
            X_k = X_k1 + alpha.unsqueeze(0) * P_k
            R_k = R_k1 - alpha.unsqueeze(0) * (A@P_k)
            R_k = R_k * A_supp
            end_iter = time.perf_counter()

            residual_norm = torch.norm(A@X_k - B, dim=1)

            if verbose:
                print("%03d | %8.4e" %
                      (k, torch.max(residual_norm/B_norm)))

            if (residual_norm <= stopping_matrix).all():
                optimal = True
                break


        end = time.perf_counter()

        if verbose:
            if optimal:
                print("Terminated in %d steps (optimal). Took %.3f ms." %
                      (k, (end - start) * 1000))
            else:
                print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                      (k, (end - start) * 1000))


        info = {
            "niter": k,
            "optimal": optimal
        }

        return X_k


    def XtX_inv(self, rho):

        if self.QQtlow is None:
            XtXInv = torch.zeros_like(self.XtX).double()
        else:
            XtXInv = self.QQtlow /rho
        
        XtXInv += self.Q_high @ torch.diag(torch.reciprocal(self.L_high + torch.ones_like(self.L_high).double()*rho)) @ self.Q_high.T
        XtXInv = XtXInv.float().to('cuda:0')
        return XtXInv
        


    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.X = None
        self.Y = None
        self.XXt = None
        self.YXt = None
        self.YtX = None
        self.XtX = None
        torch.cuda.empty_cache()
