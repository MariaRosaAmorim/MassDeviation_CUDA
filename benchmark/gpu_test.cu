#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#define pi 3.1415926535897932384626433832795028841971693993751058209

#ifndef nL
#define nL 100
#endif

#ifndef nH
#define nH 100
#endif

#ifndef nt
#define nt 1000000
#endif

#define D 2
#define Q 9

#define ERROR_FREQUENCY 1000

// Simulation tag injected by the benchmark script via -DOUTPUT_TAG="..."
// Falls back to "default" when the file is compiled standalone.
#ifndef OUTPUT_TAG
#define OUTPUT_TAG "default"
#endif

#ifndef nL
#define nL 100
#endif

#ifndef nH
#define nH 100
#endif

#ifndef R
#define R 25
#endif

__constant__ int ei[Q][D] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

__constant__ double W[Q] = {16. / 36., 4. / 36., 4. / 36., 4. / 36., 4. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
__constant__ double as;
__constant__ double cs;
__constant__ int h = 1;


#ifdef RP
__constant__ double nhp0 = 1., nhq0 = 1., rp = RP, rq = 1.;
#else
__constant__ double nhp0 = 1., nhq0 = 1., rp = 1., rq = 1.;
#endif

__constant__ double tau = 0.6;

__constant__ double beta = 0.7;       // Interface thickness parameter
__constant__ double theta_p, theta_q; // Energy deviation
__constant__ double tau_p, tau_q;

__constant__ double kappa = 0.; // Interface tension parameter

__global__ void initial_condition(double *u_x, double *u_y, double *p0, double *p, double *q, double *nh_p, double *nh_q, double *massa_h, double *nh_T)
{
    int lattice_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lattice_idx > nL * nH - 1)
        return;

    int k = lattice_idx % nL;
    int j = int(lattice_idx / nL);
    u_x[lattice_idx] = 0.;
    u_y[lattice_idx] = 0.;

    p0[lattice_idx] = 0.;

    double nhp, nhq, t2, t4, peq, qeq;

    // Static droplet
    if (pow(k - (int)(nL / 2.), 2) + pow(j - (int)(nH / 2.), 2) < pow(R, 2))
    {
        nh_p[lattice_idx] = nhp0;
        nhp = nhp0;
        nh_q[lattice_idx] = 0.;
        nhq = 0.;
    }
    else
    {
        nh_p[lattice_idx] = 0.;
        nhp = 0.;
        nh_q[lattice_idx] = nhq0;
        nhq = nhq0;
    }

    // Populations
    for (int i = 0; i < Q; i++)
    {
        int pop_idx = i + lattice_idx * Q;
        t2 = ei[i][0] * ei[i][0] + ei[i][1] * ei[i][1] - 2. * cs * cs;
        t4 = (ei[i][0] * ei[i][0] - cs * cs) * (ei[i][1] * ei[i][1] - cs * cs);

        peq = nhp * W[i] * (1. + theta_p * ((pow(as, 2) / 2.) * t2));
        peq = peq + nhp * W[i] * (pow(as, 8) / 4.) * (pow(cs, 4) * (1. - 1. / rp)) * t4;

        qeq = nhq * W[i] * (1. + theta_q * ((pow(as, 2) / 2.) * t2));
        qeq = qeq + nhq * W[i] * (pow(as, 8) / 4.) * (pow(cs, 4) * (1. - 1. / rq)) * t4;

        p[pop_idx] = peq;
        q[pop_idx] = qeq;
    }

    atomicAdd(massa_h, nhp);
    atomicAdd(nh_T, nhp + nhq);
    p0[lattice_idx] = nhp + nhq;
}

__global__ void calculate_gradients(double *u_x, double *u_y, double *nh_p, double *nh_q, double *grad_phi_x, double *grad_phi_y, double *grad_p_x, double *grad_p_y)
{
    int lattice_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lattice_idx > nL * nH - 1)
        return;

    double soma_phie_x = 0.;
    double soma_phie_y = 0.;

    double soma_pe_x = 0.;
    double soma_pe_y = 0.;

    int eix_h, eiy_h;

    int k = lattice_idx % nL;
    int j = int(lattice_idx / nL);

    double nh, rho, phi, xq, xp, nhp, nhq, ux, uy;

    for (int i = 0; i < Q; i++)
    {
        eix_h = k + (ei[i][0] * h);
        if (eix_h < 0)
        {
            eix_h = eix_h + nL;
        }
        else if (eix_h >= nL)
        {
            eix_h = eix_h - nL;
        } // Periodic consitions
        eiy_h = j + (ei[i][1] * h);
        if (eiy_h < 0)
        {
            eiy_h = eiy_h + nH;
        }
        else if (eiy_h >= nH)
        {
            eiy_h = eiy_h - nH;
        }

        int target_lattice_idx = eix_h + eiy_h * nL;
        nhp = nh_p[target_lattice_idx];
        nhq = nh_q[target_lattice_idx];
        nh = nhp + nhq;
        xp = nhp / nh;
        xq = nhq / nh;

        rho = rp * nhp + rq * nhq;

        phi = xp - xq;
        ux = u_x[target_lattice_idx];
        uy = u_y[target_lattice_idx];

        soma_phie_x = soma_phie_x + W[i] * phi * (float)ei[i][0];
        soma_phie_y = soma_phie_y + W[i] * phi * (float)ei[i][1];

        soma_pe_x = soma_pe_x + W[i] * (rho - nh) * ux * ei[i][0];
        soma_pe_y = soma_pe_y + W[i] * (rho - nh) * uy * ei[i][1];
    }

    grad_phi_x[lattice_idx] = (pow(as, 2)) * (soma_phie_x);
    grad_phi_y[lattice_idx] = (pow(as, 2)) * (soma_phie_y);

    grad_p_x[lattice_idx] = (pow(as, 2)) * (soma_pe_x);
    grad_p_y[lattice_idx] = (pow(as, 2)) * (soma_pe_y);
}

__global__ void collision_and_streaming(double *nh_p, double *nh_q, double *u_x, double *u_y, double *p, double *q, double *p_out, double *q_out, double *grad_phi_x, double *grad_phi_y, double *grad_p_x, double *grad_p_y)
{
    int lattice_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lattice_idx > nL * nH - 1)
        return;

    double ux, uy, t0, t1, t2, t3x, t3y, t4, pi_xx, pi_xy, pi_yy;
    double peq, qeq, feq, pout, qout, fout, fneq;
    double nh, rho, xq, xp, omegap, omegaq, nhp, nhq;
    double Ss, Sr, Sc;
    double taumix = tau;

    nhp = nh_p[lattice_idx];
    nhq = nh_q[lattice_idx];
    ux = u_x[lattice_idx];
    uy = u_y[lattice_idx];
    nh = nhp + nhq;
    rho = nhp * rp + nhq * rq;
    xp = nhp / nh;
    xq = nhq / nh;
    omegap = rp * nhp / rho;
    omegaq = rq * nhq / rho;

    float grad_phi_x_aux = grad_phi_x[lattice_idx];
    float grad_phi_y_aux = grad_phi_y[lattice_idx];
    float grad_phi_mod = sqrt(pow(grad_phi_x_aux, 2) + pow(grad_phi_y_aux, 2));

    float en_x = grad_phi_x_aux / grad_phi_mod;
    float en_y = grad_phi_y_aux / grad_phi_mod;

    if (grad_phi_mod < 0.000000001)
    {
        en_x = 0.;
        en_y = 0.;
    }

    pi_xx = 0.; // Second order moments
    pi_xy = 0.;
    pi_yy = 0.;
    Sc = 0.;

    taumix = xp * tau_p + xq * tau_q; // Relaxation time interpolation

    t0 = ux * ux + uy * uy;

    // Second-order non-equilibrium moment evaluation
    for (int i = 0; i < Q; i++)
    {
        int pop_idx = i + lattice_idx * Q;

        t1 = ux * ei[i][0] + uy * ei[i][1];
        t2 = ei[i][0] * ei[i][0] + ei[i][1] * ei[i][1] - 2. * cs * cs;
        t3x = ei[i][0] * (ei[i][1] * ei[i][1] - cs * cs);
        t3y = ei[i][1] * (ei[i][0] * ei[i][0] - cs * cs);
        t4 = (ei[i][0] * ei[i][0] - cs * cs) * (ei[i][1] * ei[i][1] - cs * cs);

        peq = nhp * W[i] * (1. + (pow(as, 2)) * t1 + (pow(as, 4) / 2.) * t1 * t1 - (pow(as, 2) / 2.) * t0 + theta_p * ((pow(as, 2) / 2.) * t2));
        peq = peq + nhp * W[i] * (pow(as, 6) / 2.) * (uy * (ux * ux + theta_p * cs * cs) * t3y + ux * (uy * uy + theta_p * cs * cs) * t3x);
        peq = peq + nhp * W[i] * (pow(as, 8) / 4.) * (ux * ux * uy * uy + pow(cs, 4) * (1. - 1. / rp) + theta_p * cs * cs * (ux * ux + uy * uy)) * t4;

        qeq = nhq * W[i] * (1. + (pow(as, 2)) * t1 + (pow(as, 4) / 2.) * t1 * t1 - (pow(as, 2) / 2.) * t0 + theta_q * ((pow(as, 2) / 2.) * t2));
        qeq = qeq + nhq * W[i] * (pow(as, 6) / 2.) * (uy * (ux * ux + theta_q * cs * cs) * t3y + ux * (uy * uy + theta_q * cs * cs) * t3x);
        qeq = qeq + nhq * W[i] * (pow(as, 8) / 4.) * (ux * ux * uy * uy + pow(cs, 4) * (1. - 1. / rq) + theta_q * cs * cs * (ux * ux + uy * uy)) * t4;

        pi_xx = pi_xx + (rp * (p[pop_idx] - peq) + rq * (q[pop_idx] - qeq)) * ei[i][0] * ei[i][0];
        pi_xy = pi_xy + (rp * (p[pop_idx] - peq) + rq * (q[pop_idx] - qeq)) * ei[i][0] * ei[i][1];
        pi_yy = pi_yy + (rp * (p[pop_idx] - peq) + rq * (q[pop_idx] - qeq)) * ei[i][1] * ei[i][1];
    }

    // D2Q9 second-roder non-equilibrium
    pi_xx = pi_xx;
    pi_xy = pi_xy;
    pi_yy = pi_yy;

    double eq_vel = 0., pre_vel = 0., pos_vel = 0.;

    for (int i = 0; i < Q; i++)
    {
        int pop_idx = i + lattice_idx * Q;

        t1 = ux * ei[i][0] + uy * ei[i][1];
        t2 = ei[i][0] * ei[i][0] + ei[i][1] * ei[i][1] - 2. * cs * cs;
        t3x = ei[i][0] * (ei[i][1] * ei[i][1] - cs * cs);
        t3y = ei[i][1] * (ei[i][0] * ei[i][0] - cs * cs);
        t4 = (ei[i][0] * ei[i][0] - cs * cs) * (ei[i][1] * ei[i][1] - cs * cs);

        peq = nhp * W[i] * (1. + (pow(as, 2)) * t1 + (pow(as, 4) / 2.) * t1 * t1 - (pow(as, 2) / 2.) * t0 + theta_p * ((pow(as, 2) / 2.) * t2));
        peq = peq + nhp * W[i] * (pow(as, 6) / 2.) * (uy * (ux * ux + theta_p * cs * cs) * t3y + ux * (uy * uy + theta_p * cs * cs) * t3x);
        peq = peq + nhp * W[i] * (pow(as, 8) / 4.) * (ux * ux * uy * uy + pow(cs, 4) * (1. - 1. / rp) + theta_p * cs * cs * (ux * ux + uy * uy)) * t4;

        qeq = nhq * W[i] * (1. + (pow(as, 2)) * t1 + (pow(as, 4) / 2.) * t1 * t1 - (pow(as, 2) / 2.) * t0 + theta_q * ((pow(as, 2) / 2.) * t2));
        qeq = qeq + nhq * W[i] * (pow(as, 6) / 2.) * (uy * (ux * ux + theta_q * cs * cs) * t3y + ux * (uy * uy + theta_q * cs * cs) * t3x);
        qeq = qeq + nhq * W[i] * (pow(as, 8) / 4.) * (ux * ux * uy * uy + pow(cs, 4) * (1. - 1. / rq) + theta_q * cs * cs * (ux * ux + uy * uy)) * t4;

        feq = rp * peq + rq * qeq;

        // Non-equilibrium evaluation
        fneq = W[i] * (pow(as, 4) / 2.) * (((ei[i][0] * ei[i][0] - cs * cs) * (pi_xx)) + 2. * (ei[i][0] * ei[i][1] * pi_xy) + ((ei[i][1] * ei[i][1] - cs * cs) * (pi_yy)));

        // Correction term
        Sc = 0.;
        Sc = (grad_p_x[lattice_idx] - grad_p_y[lattice_idx]) * (ei[i][0] * ei[i][0] - cs * cs);
        Sc = Sc + (grad_p_y[lattice_idx] - grad_p_x[lattice_idx]) * (ei[i][1] * ei[i][1] - cs * cs);
        Sc = W[i] * Sc * pow(as, 2) * (2. * taumix - 1) * 3. / (8. * taumix);

        // Recoloring/Segregation operator
        Sr = beta * W[i] * omegap * omegaq * ((en_x * ei[i][0]) + (en_y * ei[i][1]));

        Ss = 0.;
        // Interfacial tension source term
        Ss = ((en_x * en_x) - 1.) * (ei[i][0] * ei[i][0] - cs * cs);
        Ss = Ss + (((en_y * en_y) - 1.) * (ei[i][1] * ei[i][1] - cs * cs));
        Ss = Ss + 2. * ((en_x * en_y)) * ei[i][0] * ei[i][1];
        Ss = W[i] * kappa * Ss * nh * grad_phi_mod / (2. * (taumix)*pow(cs, 4));

        // Collision
        fout = feq + fneq * (1. - (1. / taumix));
        fout = fout + Ss + Sc;

        eq_vel = eq_vel + feq * ei[i][0];
        pre_vel = pre_vel + (rp * p[pop_idx] + rq * q[pop_idx]) * ei[i][0];
        pos_vel = pos_vel + fout * ei[i][0];

        // Segregation
        pout = omegap * fout + Sr;
        qout = omegaq * fout - Sr;

        p[pop_idx] = pout / rp;
        q[pop_idx] = qout / rq;
    }

    int eix_h, eiy_h;
    int k = lattice_idx % nL;
    int j = int(lattice_idx / nL);

    for (int i = 0; i < Q; i++)
    {
        eix_h = k + (ei[i][0] * h);
        if (eix_h < 0)
        {
            eix_h = eix_h + nL;
        }
        else if (eix_h >= nL)
        {
            eix_h = eix_h - nL;
        } // Periodic consitions
        eiy_h = j + (ei[i][1] * h);
        if (eiy_h < 0)
        {
            eiy_h = eiy_h + nH;
        }
        else if (eiy_h >= nH)
        {
            eiy_h = eiy_h - nH;
        }
        if ((eiy_h >= 0) && (eiy_h < nH))
        {
            int pop_idx = i + lattice_idx * Q;
            int pop_out_idx = i + (eix_h + eiy_h * nL) * Q;
            p_out[pop_out_idx] = p[pop_idx];
            q_out[pop_out_idx] = q[pop_idx];
        }
    }
}

__global__ void copy_streamed_pops(double *p, double *q, double *p_out, double *q_out)
{
    int lattice_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lattice_idx > nL * nH - 1)
        return;
    for (int i = 0; i < Q; i++)
    {
        int pop_idx = i + lattice_idx * Q;
        p[pop_idx] = p_out[pop_idx];
        q[pop_idx] = q_out[pop_idx];
    }
}

__global__ void update_macroscopics(double *p, double *q, double *nh_p, double *nh_q, double *u_x, double *u_y, double *nh_T, double *massa_h)
{
    int lattice_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lattice_idx > nL * nH - 1)
        return;

    double nh, nhp, nhq, ux, uy;

    nhp = 0.;
    nhq = 0.;
    ux = 0.;
    uy = 0.;

    for (int i = 0; i < Q; i++)
    {
        int pop_idx = i + lattice_idx * Q;
        nhp = nhp + p[pop_idx];
        nhq = nhq + q[pop_idx];

        ux = ux + (rp * p[pop_idx] + rq * q[pop_idx]) * ei[i][0];
        uy = uy + (rp * p[pop_idx] + rq * q[pop_idx]) * ei[i][1];
    }

    nh_p[lattice_idx] = nhp;
    nh_q[lattice_idx] = nhq;
    nh = nhp + nhq;

    u_x[lattice_idx] = ux / (rp * nhp + rq * nhq);
    u_y[lattice_idx] = uy / (rp * nhp + rq * nhq);

    atomicAdd(nh_T, nh);     // Total mass control
    atomicAdd(massa_h, nhp); // Total mass control
}

__global__ void calculate_error(double *p0, double *nh_p, double *nh_q, double *erro)
{
    int lattice_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lattice_idx > nL * nH - 1)
        return;

    double nh = 0.;
    nh = (nh_p[lattice_idx] + nh_q[lattice_idx]);
    atomicAdd(erro, pow(p0[lattice_idx] - nh, 2));
    p0[lattice_idx] = nh;
}

int main()
{
    clock_t start = clock();

    double h_kappa, h_rp, h_rq, h_tau; // Interface tension parameter
    cudaMemcpyFromSymbol(&h_tau, tau, sizeof(double));
    cudaMemcpyFromSymbol(&h_rp, rp, sizeof(double));
    cudaMemcpyFromSymbol(&h_rq, rq, sizeof(double));
    cudaMemcpyFromSymbol(&h_kappa, kappa, sizeof(double));

    double h_as = sqrt(3);
    double h_cs = 1. / h_as;
    double h_theta_p = (1. / h_rp) - 1., h_theta_q = (1. / h_rq) - 1.; // Energy deviation
    double h_tau_p = (h_tau - 0.5) * h_rp + 0.5, h_tau_q = (h_tau - 0.5) * h_rq + 0.5;

    cudaMemcpyToSymbol(as, &h_as, sizeof(double));
    cudaMemcpyToSymbol(cs, &h_cs, sizeof(double));
    cudaMemcpyToSymbol(theta_p, &h_theta_p, sizeof(double));
    cudaMemcpyToSymbol(theta_q, &h_theta_q, sizeof(double));
    cudaMemcpyToSymbol(tau_p, &h_tau_p, sizeof(double));
    cudaMemcpyToSymbol(tau_q, &h_tau_q, sizeof(double));

    int THREADS_PER_BLOCK = 64;
    int NUMBER_OF_BLOCKS = ceil(float(nL * nH) / float(THREADS_PER_BLOCK));

    static double u_x[nL * nH], u_y[nL * nH], p0[nL * nH], nh_p[nL * nH], nh_q[nL * nH];
    double massa_h, massa_h_cont, nh_T, nh_T0;
    double erro = 1., tolerancia = nL * nH * 1.e-12;
    int t = 0;

    size_t array_size = (size_t)ceil((double)nt / ERROR_FREQUENCY);
    static double *erro_arr = new double[array_size];
    static double *massa_h_arr = new double[array_size];

    const size_t macro_arr_size = nL * nH * sizeof(double);
    const size_t pop_arr_size = Q * nL * nH * sizeof(double);
    double *d_u_x, *d_u_y, *d_p0, *d_nh_q, *d_nh_p, *d_grad_phi_x, *d_grad_phi_y, *d_grad_p_x, *d_grad_p_y, *d_p_out, *d_q_out; // Device vectors
    double *d_p, *d_q;                                                                                                          // Device vectors
    double *d_massa_h, *d_nh_T, *d_erro;                                                                                        // Device vectors

    cudaMalloc((void **)&d_u_x, macro_arr_size);
    cudaMalloc((void **)&d_u_y, macro_arr_size);
    cudaMalloc((void **)&d_p0, macro_arr_size);
    cudaMalloc((void **)&d_nh_q, macro_arr_size);
    cudaMalloc((void **)&d_nh_p, macro_arr_size);
    cudaMalloc((void **)&d_grad_phi_x, macro_arr_size);
    cudaMalloc((void **)&d_grad_phi_y, macro_arr_size);
    cudaMalloc((void **)&d_grad_p_x, macro_arr_size);
    cudaMalloc((void **)&d_grad_p_y, macro_arr_size);

    cudaMalloc(&d_massa_h, sizeof(double));
    cudaMalloc(&d_nh_T, sizeof(double));
    cudaMalloc(&d_erro, sizeof(double));

    cudaMemset(d_massa_h, 0, sizeof(double));
    cudaMemset(d_nh_T, 0, sizeof(double));
    cudaMemset(d_erro, 0, sizeof(double));

    cudaMalloc((void **)&d_p, pop_arr_size);
    cudaMalloc((void **)&d_p_out, pop_arr_size);
    cudaMalloc((void **)&d_q, pop_arr_size);
    cudaMalloc((void **)&d_q_out, pop_arr_size);

    initial_condition<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_u_x, d_u_y, d_p0, d_p, d_q, d_nh_p, d_nh_q, d_massa_h, d_nh_T);

    cudaMemcpy(&massa_h, d_massa_h, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nh_T, d_nh_T, sizeof(double));

    massa_h_cont = massa_h;
    nh_T0 = nh_T;

    calculate_gradients<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_u_x, d_u_y, d_nh_p, d_nh_q, d_grad_phi_x, d_grad_phi_y, d_grad_p_x, d_grad_p_y);

    while (((erro >= tolerancia) || t < 5000) && (t < nt))
    { // Main loop (time)
        nh_T = 0.;
        collision_and_streaming<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_nh_p, d_nh_q, d_u_x, d_u_y, d_p, d_q, d_p_out, d_q_out, d_grad_phi_x, d_grad_phi_y, d_grad_p_x, d_grad_p_y);
        cudaDeviceSynchronize();
        copy_streamed_pops<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_p, d_q, d_p_out, d_q_out);
        cudaDeviceSynchronize();
        // Density and velocity update
        massa_h = 0.;
        nh_T = 0.;
        cudaMemset(d_massa_h, 0, sizeof(double));
        cudaMemset(d_nh_T, 0, sizeof(double));
        update_macroscopics<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_p, d_q, d_nh_p, d_nh_q, d_u_x, d_u_y, d_nh_T, d_massa_h);
        cudaMemcpy(&massa_h, d_massa_h, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&nh_T, d_nh_T, sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(&u_x, d_u_x, macro_arr_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(&u_y, d_u_y, macro_arr_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(&nh_p, d_nh_p, macro_arr_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(&nh_q, d_nh_q, macro_arr_size, cudaMemcpyDeviceToHost);

        // Gradient terms update
        calculate_gradients<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_u_x, d_u_y, d_nh_p, d_nh_q, d_grad_phi_x, d_grad_phi_y, d_grad_p_x, d_grad_p_y);

        if (t % ERROR_FREQUENCY == 0)
        {
            // Stop condition
            erro = 0.;
            cudaMemset(d_erro, 0, sizeof(double));
            calculate_error<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_p0, d_nh_p, d_nh_q, d_erro);
            cudaMemcpy(&erro, d_erro, sizeof(double), cudaMemcpyDeviceToHost);
            erro = sqrt(erro);
            erro_arr[t / ERROR_FREQUENCY] = erro;
            massa_h_arr[t / ERROR_FREQUENCY] = (massa_h - massa_h_cont) * 100. / massa_h_cont;
        }

        t = t + 1;
    } // Main loop end

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    FILE *filexult, *fileyult, *filerhop, *filerhoq, *file_simulation_report;
    file_simulation_report = fopen("gpu_report.txt", "w");
    filexult = fopen("ux.txt", "w");
    fileyult = fopen("uy.txt", "w");
    filerhop = fopen("rho_p.txt", "w");
    filerhoq = fopen("rho_q.txt", "w");

    // Simulation Report
    fprintf(file_simulation_report, "Tag: %s\n", OUTPUT_TAG);
    fprintf(file_simulation_report, "nL=%d nH=%d nt=%d R=%d\n", nL, nH, nt, R);
    fprintf(file_simulation_report, "Execution time: %f seconds\n", time_spent);
    fprintf(file_simulation_report, "Erro; Massa_h\n");
    for (int i = 0; i < array_size; i++)
    {
        fprintf(file_simulation_report, "%e; %e\n", (float)erro_arr[i], (float)massa_h_arr[i]);
    }

    cudaMemcpy(&u_x, d_u_x, macro_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&u_y, d_u_y, macro_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&nh_p, d_nh_p, macro_arr_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&nh_q, d_nh_q, macro_arr_size, cudaMemcpyDeviceToHost);

    for (int j = 0; j < nH; j++)
    {
        for (int k = 0; k < nL; k++)
        {
            int lattice_idx = k + j * nL;
            fprintf(fileyult, "%e\n", u_x[lattice_idx]);
            fprintf(filexult, "%e\n", u_y[lattice_idx]);
            fprintf(filerhop, "%e\n", h_rp * nh_p[lattice_idx]);
            fprintf(filerhoq, "%e\n", h_rq * nh_q[lattice_idx]);
        }
    }

    fclose(filexult);
    fclose(fileyult);
    fclose(filerhop);
    fclose(filerhoq);

    // Free device memory
    cudaFree(d_u_x);
    cudaFree(d_u_y);
    cudaFree(d_p0);
    cudaFree(d_nh_q);
    cudaFree(d_nh_p);
    cudaFree(d_grad_phi_x);
    cudaFree(d_grad_phi_y);
    cudaFree(d_grad_p_x);
    cudaFree(d_grad_p_y);

    cudaFree(d_massa_h);
    cudaFree(d_nh_T);
    cudaFree(d_erro);

    cudaFree(d_p);
    cudaFree(d_p_out);
    cudaFree(d_q);
    cudaFree(d_q_out);

    return 0;
}
