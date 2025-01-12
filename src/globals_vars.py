import datetime

today = datetime.datetime.now().strftime("%m.%d.%Y")

ct = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

sub_sequence_list = [
                        [518, 25580, 29962],                    # 0:    [INST]
                        [518, 29914, 25580, 29962],             # 1:    [/INST]
                        [3532, 14816, 29903, 6778],             # 2:    <<SYS>>
                        [529, 829, 14816, 29903, 6778],         # 3:    <</SYS>>
                        [529, 13356, 24422, 29958],             # 4:    <QUERY>
                        [1533, 13356, 24422, 29958],            # 5:    </QUERY>
                        [529, 2190, 29903, 29958],              # 6:    <ANS>
                        [1533, 2190, 29903, 29958],             # 7:    </ANS>
                        [529, 2287, 6720, 29958],               # 8:    <DEMO>
                        [1533, 2287, 6720, 29958],              # 9:    </DEMO>
                        [529, 14130, 2725, 29958],              # 10:   <QUESTION>
                        [1533, 14130, 2725, 29958],             # 11:   </QUESTION>
                        [529, 2190, 23066, 1001, 29958],        # 12:   <ANSWER>
                        [1533, 2190, 23066, 1001, 29958],       # 13:   </ANSWER>
                        [529, 23487, 29958],                    # 14:   <CASE>
                        [1533, 23487, 29958],                   # 15:   </CASE>
                        [529, 29984, 29958],                    # 16:   <Q>
                        [1533, 29984, 29958],                   # 17:   </Q>
                        [529, 29909, 29958],                    # 18:   <A>
                        [1533, 29909, 29958],                   # 19:   </A>
                        [529, 29923, 29887, 29958],             # 20:   <Eg>
                        [1533, 29923, 29887, 29958],            # 21:   </Eg>
                        [3191]                                  # 22:   _####
                    ]

custom_special_tokens = ["<QUERY>","</QUERY>","<ANS>","</ANS>","<DEMO>","</DEMO>",
                         "<QUESTION>","</QUESTION>","<ANSWER>","</ANSWER>","<CASE>","</CASE>",
                         "<Q>","</Q>","<A>","</A>","<Eg>","</Eg>",
                         "####"]

MAX_TOKEN_FOR_GSM = 400

MAX_TOKEN_FOR_MMLU = 10

MAX_TOKEN_FOR_SNI = 20

