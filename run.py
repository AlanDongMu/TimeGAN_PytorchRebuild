import numpy as np
import timegan
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
from utils import extract_time


def train(opt, ori_data):

    # Model Setting
    model = timegan.TimeGAN(opt, ori_data)
    per_print_num = opt.iterations / opt.print_times

    # 1. Embedding network training
    print('Start Embedding Network Training')
    for i in range(opt.iterations):
        model.gen_batch()
        model.batch_forward()
        model.train_embedder()
        if i % per_print_num == 0:
            print('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', e_loss: ' + str(np.round(np.sqrt(model.E_loss_T0.item()), 4)))
    print('Finish Embedding Network Training')

    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    for i in range(opt.iterations):
        model.gen_batch()
        model.batch_forward()
        model.train_supervisor()
        if i % per_print_num == 0:
            print('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', e_loss: ' + str(np.round(np.sqrt(model.G_loss_S.item()), 4)))

    # 3. Joint Training
    print('Start Joint Training')
    for i in range(opt.iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            model.gen_batch()
            model.batch_forward()
            model.train_generator(join_train=True)
            model.batch_forward()
            model.train_embedder(join_train=True)
        # Discriminator training
        model.gen_batch()
        model.batch_forward()
        model.train_discriminator()

        # Print multiple checkpoints
        if i % per_print_num == 0:
            print('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', d_loss: ' + str(np.round(model.D_loss.item(), 4)) +
                  ', g_loss_u: ' + str(np.round(model.G_loss_U.item(), 4)) +
                  ', g_loss_s: ' + str(np.round(np.sqrt(model.G_loss_S.item()), 4)) +
                  ', g_loss_v: ' + str(np.round(model.G_loss_V.item(), 4)) +
                  ', e_loss_t0: ' + str(np.round(np.sqrt(model.E_loss_T0.item()), 4)))
    print('Finish Joint Training')

    # Save trained networks
    model.save_trained_networks()


def test(opt, ori_data):

    print('Start Testing')
    # Model Setting
    model = timegan.TimeGAN(opt, ori_data)
    model.load_trained_networks()

    # Synthetic data generation
    if opt.synth_size != 0:
        synth_size = opt.synth_size
    else:
        synth_size = len(ori_data)
    generated_data = model.gen_synth_data(synth_size)
    generated_data = generated_data.cpu().detach().numpy()
    gen_data = list()
    for i in range(synth_size):
        temp = generated_data[i, :opt.seq_len, :]
        gen_data.append(temp)
    print('Finish Synthetic Data Generation')

    # Performance metrics
    metric_results = dict()
    if not opt.only_visualize_metric:
        # 1. Discriminative Score
        discriminative_score = list()
        print('Start discriminative_score_metrics')
        for i in range(opt.metric_iteration):
            print('discriminative_score iteration: ', i)
            temp_disc = discriminative_score_metrics(ori_data, gen_data)
            discriminative_score.append(temp_disc)

        metric_results['discriminative'] = np.mean(discriminative_score)
        print('Finish discriminative_score_metrics compute')

        # 2. Predictive score
        predictive_score = list()
        print('Start predictive_score_metrics')
        for i in range(opt.metric_iteration):
            print('predictive_score iteration: ', i)
            temp_predict = predictive_score_metrics(ori_data, gen_data)
            predictive_score.append(temp_predict)
        metric_results['predictive'] = np.mean(predictive_score)
        print('Finish predictive_score_metrics compute')

    # 3. Visualization (PCA and tSNE)
    visualization(ori_data, gen_data, 'pca', opt.output_dir)
    visualization(ori_data, gen_data, 'tsne', opt.output_dir)

    # Print discriminative and predictive scores
    print(metric_results)
