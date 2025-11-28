#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#define pi 3.1415926535897932384626433832795028841971693993751058209

#define nL 100
#define nH 100
#define nt 2000000
#define D 2
#define Q 9

#define R 25

// D2Q9 - Correction

int main()
{
    clock_t start = clock();

    // Vector set
    int ei[Q][D] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
    int eim[Q][D] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    // Weights
    long double W[Q] = {16. / 36., 4. / 36., 4. / 36., 4. / 36., 4. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
    // Speed of sound
    long double as = sqrt(3);
    long double cs = 1. / as;
    int h = 1;

    double nhp0 = 1., nhq0 = 1., rp = 10., rq = 1.;

    double tau = 0.6; // Relaxation time

    double beta = 0.7;                                         // Interface thickness parameter
    double theta_p = (1. / rp) - 1., theta_q = (1. / rq) - 1.; // Energy deviation
    double taumix = tau, tau_p = (tau - 0.5) * rp + 0.5, tau_q = (tau - 0.5) * rq + 0.5;

    double kappa = 0.; // Interface tension parameter

    printf("tau = %e, kappa = %e, rp = %e rq = %e\n", tau, kappa, rp, rq);

    double erro = 1., erroI = 1., tolerancia = nL * nH * 1.e-12, massa_h, massa_h_cont; // Control variables
    int t;                                                                              // time step count

    // Velocity matrix declaration and stop condition matrix
    // static double u[D][nL][nH], p0[nL][nH];
    static double u_x[nL * nH], u_y[nL * nH], p0[nL * nH];
    double ux, uy, t0, t1, t2, t3x, t3y, t4, pi_xx, pi_xy, pi_yy;

    // Density and auxiliar variables declaration
    static double nh_q[nL * nH], nh_p[nL * nH];
    double nh_T, nh_T0 = 0., nh, rho, rhop, rhoq, phi, xq, xp, omegap, omegaq, nhp, nhq;

    // Gradient terms and auxiliar variables declaration
    static double grad_phi_x[nL * nH], grad_phi_y[nL * nH], grad_p_x[nL * nH], grad_p_y[nL * nH];
    static double soma_phie_x, soma_phie_y, soma_pe_x, soma_pe_y;
    int eiy_h, eix_h;

    // Population matrixes declaration
    static double p[Q * nL * nH], q[Q * nL * nH];
    static double peq, qeq, feq, pout, qout, fout, fneq;
    static double p_out[Q * nL * nH], q_out[Q * nL * nH];
    long double Ss, Sr, Sc, fi;

    // Initial codition
    for (int j = 0; j < nH; j++)
    {
        for (int k = 0; k < nL; k++)
        {

            // Velocity field
            int lattice_idx = k + j * nL;
            u_x[lattice_idx] = 0.;
            u_y[lattice_idx] = 0.;

            p0[lattice_idx] = 0.; // Stop condition

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

            massa_h = massa_h + nhp;
            nh_T = nh_T + nhp + nhq;
        }
    }

    massa_h_cont = massa_h;
    nh_T0 = nh_T;

    // Gradient terms initialization
    for (int k = 0; k < nL; k++)
    {
        for (int j = 0; j < nH; j++)
        {

            soma_phie_x = 0.;
            soma_phie_y = 0.;

            if ((j > 0) && (j < nH - 1))
            {
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
                    }
                    eiy_h = j + (ei[i][1] * h);
                    if (eiy_h < 0)
                    {
                        eiy_h = eiy_h + nH;
                    }
                    else if (eiy_h >= nH)
                    {
                        eiy_h = eiy_h - nH;
                    }
                    nhp = nh_p[eix_h + eiy_h * nL];
                    nhq = nh_q[eix_h + eiy_h * nL];
                    nh = nhp + nhq;
                    xp = nhp / nh;
                    xq = nhq / nh;
                    phi = xp - xq;

                    soma_phie_x = soma_phie_x + W[i] * phi * (float)ei[i][0];
                    soma_phie_y = soma_phie_y + W[i] * phi * (float)ei[i][1];
                }
            }

            grad_phi_x[k + j * nL] = (pow(as, 2)) * (soma_phie_x);
            grad_phi_y[k + j * nL] = (pow(as, 2)) * (soma_phie_y);

            // grad_phi[2][k][j] = sqrt(pow(grad_phi[0][k][j], 2) + pow(grad_phi[1][k][j], 2));

            // if (grad_phi[2][k][j] < 0.000000001)
            // {
            //     en[0][k][j] = 0.;
            //     en[1][k][j] = 0.;
            // }
            // else
            // {
            //     en[0][k][j] = grad_phi[0][k][j] / grad_phi[2][k][j];
            //     en[1][k][j] = grad_phi[1][k][j] / grad_phi[2][k][j];
            // }
        }
    }

    erroI = 1.;
    t = 0;

    while (((erro >= tolerancia) || t < 5000) && (t < nt))
    { // Main loop (time)

        nh_T = 0.;

        // Collision
        for (int k = 0; k < nL; k++)
        {
            for (int j = 0; j < nH; j++)
            {
                int lattice_idx = k + j * nL;
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
                phi = xp - xq;

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

                    int pop_idx = i + lattice_idx * Q;
                    pi_xx = pi_xx + (rp * (p[pop_idx] - peq) + rq * (q[pop_idx] - qeq)) * ei[i][0] * ei[i][0];
                    pi_xy = pi_xy + (rp * (p[pop_idx] - peq) + rq * (q[pop_idx] - qeq)) * ei[i][0] * ei[i][1];
                    pi_yy = pi_yy + (rp * (p[pop_idx] - peq) + rq * (q[pop_idx] - qeq)) * ei[i][1] * ei[i][1];
                }

                // //D2Q9 second-roder non-equilibrium
                pi_xx = pi_xx;
                pi_xy = pi_xy;
                pi_yy = pi_yy;

                for (int i = 0; i < Q; i++)
                {
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

                    // Segregation
                    pout = omegap * fout + Sr;
                    qout = omegaq * fout - Sr;

                    int pop_idx = i + lattice_idx * Q;
                    p_out[pop_idx] = pout / rp;
                    q_out[pop_idx] = qout / rq;
                }
            }
        }

        // Streaming
        for (int k = 0; k < nL; k++)
        {
            for (int j = 0; j < nH; j++)
            {

                for (int i = 0; i < Q; i++)
                {
                    eix_h = k - (ei[i][0] * h);
                    if (eix_h < 0)
                    {
                        eix_h = eix_h + nL;
                    }
                    else if (eix_h >= nL)
                    {
                        eix_h = eix_h - nL;
                    } // Periodic consitions
                    eiy_h = j - (ei[i][1] * h);
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

                        int pop_idx = i + (k + j * nL) * Q;
                        int pop_out_idx = i + (eix_h + eiy_h * nL) * Q;
                        p[pop_idx] = p_out[pop_out_idx];
                        q[pop_idx] = q_out[pop_out_idx];
                    }
                }
            }
        }

        // Density and velocity update
        massa_h = 0.;
        nh_T = 0.;
        for (int k = 0; k < nL; k++)
        {
            for (int j = 0; j < nH; j++)
            {
                nhp = 0.;
                nhq = 0.;
                ux = 0.;
                uy = 0.;
                int lattice_idx = k + j * nL;

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

                nh_T = nh_T + nh;        // Total mass control
                massa_h = massa_h + nhp; // Heavier component mass control
            }
        }

        // Gradient terms update
        for (int k = 0; k < nL; k++)
        {
            for (int j = 0; j < nH; j++)
            {

                soma_phie_x = 0.;
                soma_phie_y = 0.;

                soma_pe_x = 0.;
                soma_pe_y = 0.;

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

                    int lattice_idx = eix_h + eiy_h * nL;
                    nhp = nh_p[lattice_idx];
                    nhq = nh_q[lattice_idx];
                    nh = nhp + nhq;
                    xp = nhp / nh;
                    xq = nhq / nh;

                    rho = rp * nhp + rq * nhq;

                    phi = xp - xq;
                    ux = u_x[lattice_idx];
                    uy = u_y[lattice_idx];

                    soma_phie_x = soma_phie_x + W[i] * phi * (float)ei[i][0];
                    soma_phie_y = soma_phie_y + W[i] * phi * (float)ei[i][1];
                    // soma_phie_x = soma_phie_x + W[i]*phi*(6.-(float)(ei[i][0]*ei[i][0]+ei[i][1]*ei[i][1])*as*as)*(float)ei[i][0];
                    // soma_phie_y = soma_phie_y + W[i]*phi*(6.-(float)(ei[i][0]*ei[i][0]+ei[i][1]*ei[i][1])*as*as)*(float)ei[i][1];

                    soma_pe_x = soma_pe_x + W[i] * (rho - nh) * ux * ei[i][0];
                    soma_pe_y = soma_pe_y + W[i] * (rho - nh) * uy * ei[i][1];
                    // soma_pe_x = soma_pe_x + W[i]*(rho-nh)*ux*(6.-(float)(ei[i][0]*ei[i][0]+ei[i][1]*ei[i][1])*as*as)*(float)ei[i][0];
                    // soma_pe_y = soma_pe_y + W[i]*(rho-nh)*uy*(6.-(float)(ei[i][0]*ei[i][0]+ei[i][1]*ei[i][1])*as*as)*(float)ei[i][1];
                }

                int lattice_idx = k + j * nL;
                grad_phi_x[lattice_idx] = (pow(as, 2)) * (soma_phie_x);
                grad_phi_y[lattice_idx] = (pow(as, 2)) * (soma_phie_y);

                // grad_phi[2][k][j] = sqrt(pow(grad_phi[0][k][j], 2) + pow(grad_phi[1][k][j], 2));

                // if (grad_phi[2][k][j] == 0.000000)
                // {
                //     en[0][k][j] = 0.;
                //     en[1][k][j] = 0.;
                // }
                // else
                // {
                //     en[0][k][j] = grad_phi[0][k][j] / grad_phi[2][k][j];
                //     en[1][k][j] = grad_phi[1][k][j] / grad_phi[2][k][j];
                // }
                grad_p_x[lattice_idx] = (pow(as, 2)) * (soma_pe_x);
                grad_p_y[lattice_idx] = (pow(as, 2)) * (soma_pe_y);
            }
        }

        if (t % 1000 == 0)
        {
            // Stop condition
            erro = 0.;
            for (int k = 0; k < nL; k++)
            {
                for (int j = 0; j < nH; j++)
                {
                    int lattice_idx = k + j * nL;
                    nh = (nh_p[lattice_idx] + nh_q[lattice_idx]);
                    erro = erro + pow(p0[lattice_idx] - nh, 2);
                    p0[lattice_idx] = nh;
                }
            }
            erro = sqrt(erro);
            printf(" t =  %d;   erro = %e,  porcent_dev_T = %e  \n", t, (float)erro, (float)(massa_h - massa_h_cont) * 100. / massa_h_cont);
        }

        t = t + 1;
    } // Main loop end

    FILE *filexult, *fileyult, *filerhop, *filerhoq;
    filexult = fopen("ux.txt", "w");
    fileyult = fopen("uy.txt", "w");
    filerhop = fopen("rho_p.txt", "w");
    filerhoq = fopen("rho_q.txt", "w");

    for (int j = 0; j < nH; j++)
    {
        for (int k = 0; k < nL; k++)
        {
            int lattice_idx = k + j * nL;
            fprintf(fileyult, "%e\n", u_x[lattice_idx]);
            fprintf(filexult, "%e\n", u_y[lattice_idx]);
            fprintf(filerhop, "%e\n", rp * nh_p[lattice_idx]);
            fprintf(filerhoq, "%e\n", rq * nh_q[lattice_idx]);
        }
    }

    fclose(filexult);
    fclose(fileyult);
    fclose(filerhop);
    fclose(filerhoq);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", time_spent);

    return 0;
}
