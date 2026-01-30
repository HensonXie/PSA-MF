import torch
from torch import nn
from modules.encoders import RNNEncoder, SubNet, LanguageEmbeddingLayer, PLanguageEmbeddingLayer, ssLanguageEmbeddingLayer, cosine_similarity_loss, Clip

class PSAMF(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        
        # Encoders
        self.text_enc = LanguageEmbeddingLayer(hp)
        self.p_text_enc = PLanguageEmbeddingLayer(hp)
        self.ss_text_enc = ssLanguageEmbeddingLayer(hp)
        self.visual_enc = RNNEncoder(
            in_size=hp.d_vin, #20
            hidden_size=hp.d_vh, #64
            out_size=hp.d_vout,  #64
            num_layers=hp.n_layer, #1
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size=hp.d_ain, #5
            hidden_size=hp.d_ah, 
            out_size=hp.d_aout, 
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        self.fusion_prj = SubNet(
            in_size=190,
            hidden_size=hp.d_prjh, 
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        
        self.n = 64
        self.clip = Clip(self.n, self.n)
        self.text_projection = nn.Linear(768, self.n)
        self.text_projection2 = nn.Linear(768, self.n)
        self.personality_projection = nn.Linear(768, self.n)
        self.y_linear = nn.Linear(self.n, 1)
        self.l1loss = nn.L1Loss()
        
        self.attention_v = nn.MultiheadAttention(hp.d_vout, 2)
        self.attention_a = nn.MultiheadAttention(hp.d_vout, 2)
        self.r_v = nn.Linear(768, hp.d_vout)
        self.r_a = nn.Linear(768, hp.d_vout)
        
        self.v_dim_v = nn.Linear(hp.d_vout, 2048)
        self.a_dim_v = nn.Linear(hp.d_vout, 2048)
        self.layer_norm = nn.LayerNorm(4096 + hp.d_vout)
        self.ys = nn.Linear(4096 + hp.d_vout, 128)
        self.cnn = nn.Conv1d(3, 1, 3)

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, p_bert_sentences=None, p_bert_sentence_types=None, p_bert_sentence_att_mask=None):
        # Audio and visual encoding
        acoustic_e, aco_rnn_output = self.acoustic_enc(acoustic, a_len, 500)# (218,256,5)->(256,64),(500,256,64)
        visual_e, vis_rnn_output = self.visual_enc(visual, v_len, 500) # (261,256,20)->(256,64),(500,256,64)

        # Extract BERT hidden states for layers 0-11
        with torch.no_grad():
            p_text_hidden, p_hidden_states = self.p_text_enc(sentences, p_bert_sentences, p_bert_sentence_types, p_bert_sentence_att_mask)
        
        vector2 = torch.ones((bert_sent_mask.shape[0], 2), device=bert_sent_mask.device)# （256，2）
        p_hidden_statess = [p_hidden_states, acoustic_e, visual_e]
        att_mask = bert_sent_mask
        att_mask2 = torch.cat([bert_sent_mask, vector2], dim=1)
        att_mask = [att_mask, att_mask2]
        last_lav_hidden_state, lav_hidden_statess, distill_loss_6 = self.ss_text_enc(sentences, bert_sent, bert_sent_type, att_mask, p_hidden_statess)
        se_text_hidden, se_hidden_states = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask)
        
        align_loss = distill_loss_6[1]
        lav_text = last_lav_hidden_state[:, 0, :]
        pe_text = p_text_hidden[:, 0, :]
        
        # Iterate over each BERT layer output and calculate NCE and y_loss
        nce_sen_list = []
        y_loss_sen_list = []

        for i, sentiment_layer in enumerate(lav_hidden_statess):
            sentiment = sentiment_layer[:, 0, :]
        
            sen_p1 = self.text_projection(sentiment)
            sen_p2 = self.text_projection2(sentiment)
            personality_proj = self.personality_projection(pe_text)

            nce_sen = self.clip(sen_p1, personality_proj)
            sim_sen = cosine_similarity_loss(sen_p1, personality_proj)
            nce_sen *= sim_sen #0.几左右

            nce_sen_list.append(nce_sen)

            if y is not None:
                y_pred = self.y_linear(sen_p2)
                y_loss_sen = self.l1loss(y_pred, y) * (1 - sim_sen)#1.4左右
                y_loss_sen_list.append(y_loss_sen)
            else:
                y_loss_sen_list.append(torch.tensor([0.0]))

                # Attention mechanisms
        t_v = self.r_v(lav_text).unsqueeze(1).permute(1, 0, 2)# (256,768)->(256,64)->(1,256,64)
        t_vv = self.attention_v(t_v, vis_rnn_output, vis_rnn_output)[0].squeeze(0)
        t_aa = self.attention_a(t_v, aco_rnn_output, aco_rnn_output)[0].squeeze(0)
        
        # Feature transformations
        vv = self.v_dim_v(t_vv)#（256，64）->（256，2048）
        aa = self.a_dim_v(t_aa)
        
        T_all_tav = self.ys(self.layer_norm(torch.cat([vv, aa, t_v.permute(1, 0, 2).squeeze(1)], dim=1)))
        C_all_tav = self.cnn(torch.cat([t_vv.unsqueeze(1), t_aa.unsqueeze(1), t_v.permute(1, 0, 2)], dim=1)).squeeze(1)
        
        # Final fusion
        full_feature = torch.cat([C_all_tav, T_all_tav], dim=1)
        fusion, preds = self.fusion_prj(full_feature)

        return nce_sen_list, preds, y_loss_sen_list, align_loss