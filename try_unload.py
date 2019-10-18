from emnist import extract_training_samples
from emnist import extract_test_samples
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import mnist as ms

def makes_one_hot_vectors(array):
    temp = array
    one_hot_vector = np.zeros((temp.size,temp.max()+1))
    one_hot_vector[np.arange(temp.size),temp] = 1
    return one_hot_vector

    # rescale np.array from uint8 to float32
def rescale_to_float32(array):
    return array.astype(np.float32) / 255

    # Normalize images
def normalize_image(image):
    temp_img = image.reshape([28,28]).copy()
    flip_x = np.fliplr(temp_img).copy()
    rotate90_aCW = np.rot90(flip_x).copy()
    fixed_image = rotate90_aCW.copy()
    normalized_image = fixed_image.reshape([1,784]).copy()
    return normalized_image

''' ################################################   FROM GOOGLE COLAB   #########################################'''

def xavier_init(shape):
    return tf.random_normal(shape = shape, stddev = 1./tf.sqrt(shape[0]/2.0))

def start_everything():
    ''' ######################## SECTION 1 ##############################'''

    # Training parameters

    learning_rate = 0.0002
    batch_size = 128 # from (128)
    epochs = 100000 # from (100K,200K)

    # Network parameters
    image_dim = 784 # because image size is 28x28, hence 784

    Y_dimension = 27 # labels dimensions

    gen_hidd_dim = 256
    disc_hidd_dim = 256
    z_noise_dim = 100 # Imput noise datapoint size 100x1

    ''' ######################## SECTION 2 ##############################'''
    # Define weights and bias dictionaries

    weights = {"disc_H"     : tf.Variable(xavier_init([image_dim + Y_dimension,disc_hidd_dim])), # concat Y_dimension to discriminator inputs
               "disc_final" : tf.Variable(xavier_init([disc_hidd_dim, 1])),
               "gen_H"      : tf.Variable(xavier_init([z_noise_dim + Y_dimension, gen_hidd_dim])), # concat Y_dimension to generator noise inputs
               "gen_final"  : tf.Variable(xavier_init([gen_hidd_dim,image_dim]))   
              }

    bias =    {"disc_H"     : tf.Variable(xavier_init([disc_hidd_dim])),
               "disc_final" : tf.Variable(xavier_init([1])),
               "gen_H"      : tf.Variable(xavier_init([gen_hidd_dim])),
               "gen_final"  : tf.Variable(xavier_init([image_dim]))   
              }

    # Create the computational graph

    # Define the placeholders for External input
    X_input = tf.placeholder(tf.float32, shape = [None, image_dim], name = "real_input")
    Y_input = tf.placeholder(tf.float32, shape = [None, Y_dimension], name = "Labels")
    Z_input = tf.placeholder(tf.float32, shape = [None, z_noise_dim], name = "input_noise")
    

    # Define Discriminator function
    def Discriminator(x,y):
        inputs = tf.concat(axis = 1, values = [x,y])
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(inputs, weights["disc_H"]), bias["disc_H"]))
        final_layer = tf.add(tf.matmul(hidden_layer, weights["disc_final"]), bias["disc_final"])
        disc_output = tf.nn.sigmoid(final_layer)
        return final_layer, disc_output

    # Define Generator function
    def Generator(x,y):
        inputs = tf.concat(axis = 1, values = [x,y])
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(inputs, weights["gen_H"]), bias["gen_H"]))
        final_layer = tf.add(tf.matmul(hidden_layer, weights["gen_final"]), bias["gen_final"])
        gen_output = tf.nn.sigmoid(final_layer)
        return gen_output

    # Building the Generator Network
    with tf.name_scope("Generator") as scope:
        output_Gen = Generator(Z_input, Y_input)
      
    # Building Discriminator Network
    with tf.name_scope("Discriminator") as scope:
        real_output1_Disc, real_output_Disc = Discriminator(X_input,Y_input)     # Implements D(x)
        fake_output1_Disc, fake_output_Disc = Discriminator(output_Gen,Y_input)  # Implemnets D(G(x))
        

    ''' ######################## SECTION 3 ##############################'''
    # First kind of loss
##    with tf.name_scope("Discriminator_Loss") as scope:
##        Discriminator_Loss = -tf.reduce_mean(tf.log(real_output_Disc + 0.0001) + tf.log(1.- fake_output_Disc + 0.0001))
##      
##    with tf.name_scope("Generator_Loss") as scope:
##        Generator_Loss = -tf.reduce_mean(tf.log(fake_output_Disc + 0.0001)) # due to max log(D(G(z)))
##      
##    # TensorBoard Summary
##    Disc_loss_total = tf.summary.scalar("Disc_Total_loss", Discriminator_Loss)
##    Gen_loss_total = tf.summary.scalar("Gen_Loss", Generator_Loss)


    ''' ######################## SECTION 4 ##############################'''
    # Second type of loss

    with tf.name_scope("Discriminator_Loss") as scope:
        Disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = real_output1_Disc, labels = tf.ones_like(real_output1_Disc)))
        Disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_output1_Disc, labels = tf.zeros_like(fake_output1_Disc)))
        Discriminator_Loss = Disc_real_loss + Disc_fake_loss
      
    with tf.name_scope("Generator_Loss") as scope:
        Generator_Loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_output1_Disc, labels = tf.ones_like(fake_output1_Disc)))
      
    # TensorBorad Summary
    Disc_loss_real_summary = tf.summary.scalar("Disc_loss_real", Disc_real_loss)
    Disc_loss_fake_summary = tf.summary.scalar("Disc_loss_fake", Disc_fake_loss)
    Disc_loss_summary = tf.summary.scalar("Disc_Total_loss", Discriminator_Loss)

    Disc_loss_total = tf.summary.merge([Disc_loss_real_summary, Disc_loss_fake_summary, Disc_loss_summary])
    Gen_loss_total = tf.summary.scalar("Gen_Loss", Generator_Loss)


    ''' ######################## SECTION 5 ##############################'''
    # Define the variables 
    Generator_var = [weights["gen_H"], weights["gen_final"], bias["gen_H"], bias["gen_final"]]
    Discriminator_var = [weights["disc_H"], weights["disc_final"], bias["disc_H"], bias["disc_final"]]

    # Define the optimizer

    with tf.name_scope("Optimizer_Discriminator") as scope:
        Discriminator_optimize = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(Discriminator_Loss, var_list = Discriminator_var)
      
    with tf.name_scope("Optimizer_Generator") as scope:
        Generator_optimize = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(Generator_Loss, var_list = Generator_var)


    ''' ########################  SECTION 6 (Main runing code block) ##############################'''
    # Initialize the variables
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    writer = tf.summary.FileWriter("./log", sess.graph)

    for epoch in range(epochs):
        X_batch, Y_label = emnist.train.next_batch(batch_size)      

      # Generate noise to feed the discriminator 
        Z_noise = np.random.uniform(-1.,1.,size = [batch_size, z_noise_dim])
        _, Disc_loss_epoch = sess.run([Discriminator_optimize, Discriminator_Loss], feed_dict = {X_input:X_batch, Y_input:Y_label, Z_input:Z_noise})
        _, Gen_loss_epoch = sess.run([Generator_optimize, Generator_Loss], feed_dict = {Z_input: Z_noise, Y_input:Y_label})

        # Running the Discriminator summary
        summary_Disc_Loss = sess.run(Disc_loss_total, feed_dict = {X_input:X_batch, Y_input:Y_label, Z_input:Z_noise})
        # Adding the Discriminator summary
        writer.add_summary(summary_Disc_Loss, epoch)

        # Running the Generator summary 
        summary_Gen_Loss = sess.run(Gen_loss_total, feed_dict = {Z_input:Z_noise, Y_input:Y_label})
        # Adding the Discriminator summary
        writer.add_summary(summary_Gen_Loss, epoch)

        if epoch % 2000 == 0:
            print ("Steps :{0} : Generator Loss :{1}, Discriminator Loss :{2}".format(epoch,Gen_loss_epoch,Disc_loss_epoch))
            

    ''' ######################## SECTION 7 ##############################'''
    # Testing
    # Generate images from noise, using the generator network.

##    n = 6
##
##    for i in range(10):
##        dateTimeObj = datetime.now()
##        dateNtime = dateTimeObj.strftime("%d-%b-%Y(%H-%M-%S)") 
##        
##        canvas = np.empty((28 * n, 28 * n))
##        for i in range(n):
##            # Noise input
##            Z_noise = np.random.uniform(-1.,1., size = [batch_size, z_noise_dim])
##
##            # Generate image from noise.
##            g = sess.run(output_Gen, feed_dict = {Z_input: Z_noise})
##
##            # Reverse colours for better display 
##            g = -1 * (g-1)
##            for j in range(n):
##                # Draw the generated digits
##                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28,28])
##            
##        plt.figure(figsize=(n,n))
##        plt.imshow(canvas, origin="upper", cmap="gray")
##        plt.savefig(dateNtime + '.png', bbox_inches="tight")
##        plt.show()

    def generate_plot(samples):
        fig = plt.figure(figsize = (4,4))
        gs = gridspec.GridSpec(4,4)
        gs.update(wspace = 0.05, hspace = 0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticks([])
            ax.set_yticks({})
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28,28)*-1, cmap = 'gray')
        return fig
    
    def create(inp):
        feature_map = {"a":1,
                       "b":2,
                       "c":3,
                       "d":4,
                       "e":5,
                       "f":6,
                       "g":7,
                       "h":8,
                       "i":9,
                       "j":10,
                       "k":11,
                       "l":12,
                       "m":13,
                       "n":14,
                       "o":15,
                       "p":16,
                       "q":17,
                       "r":18,
                       "s":19,
                       "t":20,
                       "u":21,
                       "v":22,
                       "w":23,
                       "x":24,
                       "y":25,
                       "z":26
                        }
        # Number of samples to be displayed for each category
        samples = 16

        Z_noise = np.random.uniform(-1.,1., size = [samples, z_noise_dim])

        # Createone-hot label vector
        Y_label = np.zeros(shape = [samples, Y_dimension])
        Y_label[:, feature_map[inp]] = 1

        # run the trained Generator excluding discriminator
        generated_samples = sess.run(output_Gen, feed_dict = {Z_input:Z_noise, Y_input: Y_label})

        # Plotting & saving the images
        dateTimeObj = datetime.now()
        dateNtime = dateTimeObj.strftime("%d-%b-%Y(%H-%M-%S)")
        fig = generate_plot(generated_samples)
        plt.savefig(inp + '__' + dateNtime + '.png', bbox_inches = 'tight') # Saving the figures
        
    
    create('y')
    create('e')
    create('s')


if __name__ == '__main__':
    emnist = ms.read_data_sets("emnist_letters/",one_hot = True)
    ##n = 6
    ##sample_image = emnist.train.next_batch(n)
    ##
    ##create_results(sample_image, 6)
##    init = 30
##    for i in range(15):
##        plt.figure(figsize=(1,1))
##        plt.imshow(emnist.train.images[init+i].reshape([28,28])*-1, cmap="gray")
##        plt.savefig('tete.png', bbox_inches = 'tight')
##        print (np.where(emnist.train.labels[init+i] == 1.))
##        plt.show()

    start_everything()
