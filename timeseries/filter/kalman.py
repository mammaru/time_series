#coding: utf-8
"""
Kalman's algorithm(prediction, filtering, smoothing)

Author: mammaru <mauma1989@gmail.com>

"""




if __name__ == "__main__":
    print "kalman.py: Called in main process."

    if 0:
        obs_p = 2
        sys_k = 2
        N = 20
        kl = Kalman(obs_p,sys_k)
        data = kl.ssm.gen_data(N)
        #kl.set_data(data[1])
        #kl.pfs()
        #for i in range(100): kl.pfs()
        em = EM(data[1], sys_k)
        em.execute()

        if 1:
            fig, axes = plt.subplots(np.int(np.ceil(sys_k/3.0)), 3, sharex=True)
            j = 0
            for i in range(3):
                while j<sys_k:
                    if sys_k<=3:
                        axes[j%3].plot(data[0][j], "k--", label="obs")
                        axes[j%3].plot(em.kl.xp[j], label="prd")
                        axes[j%3].legend(loc="best")
                        axes[j%3].set_title(j)
                    else:
                        axes[i, j%3].plot(data[0][j], "k--", label="obs")
                        axes[i, j%3].plot(em.kl.xp[j], label="prd")
                        axes[i, j%3].legend(loc="best")
                        axes[i, j%3].set_title(j)
                    j += 1
                    if j%3 == 2: break

            fig.show()
        
        #loss = data[0]-em.kl.xs
        #plt.plot(loss)
        #plt.plot(em.llh)
        #plt.show()