import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class Encoderlstm(Layer):
    def __init__(self, m):
        """
        m : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Encoderlstm, self).__init__(name="encoder_lstm")
        self.lstm = LSTM(m, return_state=True)
        self.initial_state = None

    def call(self, x, training=False):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]


class InputAttention(Layer):
    def __init__(self, T):
        super(InputAttention, self).__init__(name="input_attention")
        self.w1 = Dense(T)
        self.w2 = Dense(T)
        self.v = Dense(1)

    def call(self, h_s, c_s, x):
        """
        h_s : hidden_state (shape = batch,m)
        c_s : cell_state (shape = batch,m)
        x : time series encoder inputs (shape = batch,T,n)
        """
        # 논문에서 hidden과 cell state를 concat하여 어텐션하는게 정확도가 더 높다고 한다.
        query = tf.concat([h_s, c_s], axis=-1)  # batch, m*2
        # 특징 개수만큼 hidden과 cell 합친 상태를 불린다. (각 특징만큼 hidden과 cell의 어텐션을 보기 위함이니까.)
        query = RepeatVector(tf.shape(x)[2])(query)  # batch, n, m*2
        # 2열과 1열 전치
        x_perm = Permute((2, 1))(x)  # batch, n, T

        # 일반적인 바다나우 어텐션 실시.
        score = tf.nn.tanh(self.w1(x_perm) + self.w2(query))  # batch, n, T

        # 1 시계열에 대한, 특징들의 scoring이 진행된다.
        score = self.v(score)  # batch, n, 1
        
        # Attention Weight가 1 시계열에 대한 특징들의 어텐션이었으니, 소스에서 인지하기 쉽게 2열 1열 다시 전치
        score = Permute((2, 1))(score)  # batch,1,n

        attention_weights = tf.nn.softmax(score)  # t 번째 time step 일 때 각 feature 별 중요도
        return attention_weights


class Encoder(Layer):
    def __init__(self, T, m):
        super(Encoder, self).__init__(name="encoder")
        """
        m : 인코더 LSTM의 Units(outputs) 개수. (lstm의 출력을 몇개의 차원으로 할 것인가?)
        """
        self.T = T
        self.input_att = InputAttention(T)
        self.lstm = Encoderlstm(m)
        self.initial_state = None
        self.alpha_t = None

    def call(self, data, h0, c0, n, training=False):
        """
        data : encoder data (shape = batch, T, n)
        n : data feature num
        """
        # state setting
        self.lstm.reset_state(h0=h0, c0=c0)

        # 시간별 result를 담을 array 생성
        alpha_seq = tf.TensorArray(tf.float32, self.T)
        
        # 이부분은 각자 데이터셋에 따라 수정이 필요할 수 있다.
        # 근데, 사실상 encoder는 집어넣은 dataset의 timestep을 다 쓸테니, 수정이 필요하지 않는게 일반적일듯 하다.
        for t in range(self.T):

            # 배치단위로 각 시간순서의 값들을 떼어서 3차원으로 변환시켜준다
            # 24시간으로 가정하면, 1배치의 모든 0시 값을 t=0에 수행
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]
            # x : (batch_size,1,n)
            
            # 각 배치의 모든 시 1 time을 넣어서 lstm을 수행한다. (1 time step lstm)
            h_s, c_s = self.lstm(x)
            # h_s, c_s : (batch_size, m)

            # 1 time step에 대한 input attetnion을 진행한다.
            self.alpha_t = self.input_att(h_s, c_s, data)  # batch,1,n
            # 한 시계열에 대한 feature attention score가 출력된다.

            # 이후에 곱을 한꺼번에 진행하기 위해서 value와 곱하는 과정을 먼저 수행하진 않고 결과 array에 전부 넣어놓는다.
            alpha_seq = alpha_seq.write(t, self.alpha_t)

        # 각 시계열별로 특징들에 대한 weight를 전부 계산하고, array를 1개로 합쳐 최종 시계열별 특징 중요도에 대한 array set를 완성한다.
        alpha_seq = tf.reshape(alpha_seq.stack(), (-1, self.T, n))  # batch, T, n

        # Attention Weight * Value를 통해, 실제 Context_Vector set를 구하게 된다.
        # 결과적으로는 각 시간 time-line별로 가장 중요하게 생각되는 특성들만 값이 높아지게 된다.
        output = tf.multiply(data, alpha_seq)  # batch, T, n

        return output


class Decoderlstm(Layer):
    def __init__(self, p):
        """
        p : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Decoderlstm, self).__init__(name="decoder_lstm")
        self.lstm = LSTM(p, return_state=True)
        self.initial_state = None

    def call(self, x, training=False):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]


class TemporalAttention(Layer):
    def __init__(self, m):
        super(TemporalAttention, self).__init__(name="temporal_attention")
        self.w1 = Dense(m)
        self.w2 = Dense(m)
        self.v = Dense(1)

    def call(self, h_s, c_s, enc_h):
        """
        h_s : hidden_state (shape = batch,p)
        c_s : cell_state (shape = batch,p)
        enc_h : time series encoder inputs (shape = batch,T,m)
        """
        # 평범한 바다나우 어텐션이 진행된다.
        query = tf.concat([h_s, c_s], axis=-1)  # batch, p*2
        # 여기서는 시계열만큼을 복사한다는데 유의하자. decoder의 temporal attention은 일반적인 바다나우와 같이, 시간에 대한 attention이 계산된다.
        query = RepeatVector(tf.shape(enc_h)[1])(query)
        score = tf.nn.tanh(self.w1(enc_h) + self.w2(query))  # batch, T, m
        score = self.v(score)  # batch, T, 1
        attention_weights = tf.nn.softmax(
            score, axis=1
        )  # encoder hidden state h(i) 의 중요성 (0<=i<=T)
        return attention_weights


class Decoder(Layer):
    def __init__(self, T, p, m):
        super(Decoder, self).__init__(name="decoder")
        self.T = T
        self.temp_att = TemporalAttention(m)
        self.dense = Dense(1)
        self.lstm = Decoderlstm(p)
        self.enc_lstm_dim = m
        self.dec_lstm_dim = p
        self.context_v = None
        self.dec_h_s = None
        self.beta_t = None

    def call(self, data, enc_h, h0=None, c0=None, training=False):
        """
        data : decoder data
        enc_h : encoder hidden state (shape = batch, T, m)
        """

        h_s = None
        self.lstm.reset_state(h0=h0, c0=c0)

        self.context_v = tf.zeros((tf.shape(enc_h)[0], 1, self.enc_lstm_dim))  # batch,1,m
        self.dec_h_s = tf.zeros((tf.shape(enc_h)[0], self.dec_lstm_dim))  # batch, p
        # 이부분은 각자 데이터셋에 따라 수정이 필요할 수 있다.
        # 필자는 이것저것 시도해봤었는데, 이부분은 decoder time step 전체를 보고 어떤 값 1개를 예측하기 위함이었다.
        for t in range(data.shape[1]):
            # decoder에 집어넣은 data를 모든 배치별 1 time step으로 분리한다.
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]  #  (batch,1,특징값 혹은 예측할값 기타등등?<-구성하기 나름이겠다.)

            # 바다나우 어텐션과 동일하게, x와 context_vector를 concat
            x = tf.concat([x, self.context_v], axis=-1) # batch, 1, m+1
            x = self.dense(x)  # batch,1,특징 갯수 혹은 예측해야할 갯수+context_vector size

            # attention을 진행할 decoder의 hidden과 cell을 구한다.
            h_s, c_s = self.lstm(x)  # batch,p


            # encoding에서 weighted sum된 특징값들을 통해서 (특징 중요도로 학습한 결과로), decoder에서는 시간에 대한 attention이 진행되게 된다.
            self.beta_t = self.temp_att(h_s, c_s, enc_h)  # batch, T, 1
            # 결과는 시간들에 있어서 가장 중요한 값이 뭐였는지에 대한 weigth가 나온다

            # 실제로 encoder 시간값들(사실상 value)에 곱해서 가장 유의미했던 시간값들의 특징값에는 더 많은 가중치가 곱해지게 된다.
            self.context_v = tf.matmul(
                self.beta_t, enc_h, transpose_a=True
            )  # batch,1,m

        return tf.concat(
            [h_s[:, tf.newaxis, :], self.context_v], axis=-1
        )  # batch,1,m+p


class DARNN(Model):
    def __init__(self, T, m, p, target_len):
        super(DARNN, self).__init__(name="DARNN")
        """
        T : time step (24시간이면 24)
        m : encoder lstm feature(output) length (정확히는 encoder lstm의 유닛개수)
        p : decoder lstm feature(output) length (정확히는 decoder lstm의 유닛개수)
        h0 : lstm hidden state
        c0 : lstm cell state
        target_len : 예측할 Y의 개수 (24시간에 대한 기온,풍속,기압 을 예측할거면 target_len=3)
        """
        self.m = m
        self.encoder = Encoder(T=T, m=m)
        self.decoder = Decoder(T=T, p=p, m=m)
        self.lstm = LSTM(m, return_sequences=True)
        self.dense1 = Dense(p)
        self.dense2 = Dense(target_len)

    def call(self, inputs, training=False, mask=None):
        """
        inputs : [enc , dec]
        enc_data : batch,T,n
        dec_data : batch,T-1,1
        """
        
        enc_data, dec_data = inputs
        batch = tf.shape(enc_data)[0]

        '''
        h0, c0는 솔직히 요즘 keras에서 초기화가 꼭 필요한지 궁금하다.
        공식 개발자 가이드를 찾아봐도, 초기 h0과 c0은 0 dimension으로 채워진다고 되어있다.
        '''
        h0 = tf.zeros((batch, self.m))
        c0 = tf.zeros((batch, self.m))
        
        # input attention을 이용하여, 각 시계열별 특징 중요도를 한땀한땀 계산하여 output에 해당하는 dimension으로 출력한다.
        enc_output = self.encoder(
            enc_data, n=tf.shape(enc_data)[2], h0=h0, c0=c0, training=training
        )  # output : (batch_size, T, n)

        # 전부 재계산된 context_vector를 통해서, 전 시계열을 1회 학습한다.
        enc_h = self.lstm(enc_output)  # batch, T, m

        # temporal attention을 이용하여, enc 시계열 대비 dec 시계열이 어떤 시계열에 attention되는지를 계산한다.
        dec_output = self.decoder(
            dec_data, enc_h, h0=h0, c0=c0, training=training
        )  # output : (batch_size,1,m+p)
        
        # 전체 특징단위, 시간단위 유의미한 값들이 가중된 어떤 벡터값이 나오니, 그냥 NN 시킨다.
        output = self.dense2(self.dense1(dec_output))
        output = tf.squeeze(output)
        return output
