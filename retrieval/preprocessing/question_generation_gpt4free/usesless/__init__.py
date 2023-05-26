import json
import requests
import sys


class Completion:
    headers = {
        "authority": "ai.usesless.com",
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.5",
        "cache-control": "no-cache",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0",
    }

    @staticmethod
    def create(
        systemMessage: str = "You are a helpful assistant",
        prompt: str = "",
        parentMessageId: str = "",
        presence_penalty: float = 1,
        temperature: float = 1,
        model: str = "gpt-3.5-turbo",
        proxy_https : str = 'empty',
    ):
        # print(parentMessageId, prompt)

        json_data = {
            "openaiKey": "",
            "prompt": prompt,
            "options": {
                "parentMessageId": parentMessageId,
                "systemMessage": systemMessage,
                "completionParams": {
                    "presence_penalty": presence_penalty,
                    "temperature": temperature,
                    "model": model,
                },
            },
        }



        url = "https://ai.usesless.com/api/chat-process"

        # added 11.05
        
        limit = 1
        ii = 0
        while True:
            
            proxies= {
                    "https": proxy_https
                }
            if ii >= limit:
                return None
            try:
                ii = ii + 1
                # print(proxy_https)
                request = requests.post(url, headers=Completion.headers, json=json_data, proxies=proxies, timeout=10, verify=False)
                break
            except KeyboardInterrupt:
                # quit
                sys.exit()
            except Exception as e:
                pass
                
                

        # ---------
        # print("success")
        content = request.content

        response = Completion.__response_to_json(content)
        return response

    @classmethod
    def __response_to_json(cls, text) -> dict:
        text = str(text.decode("utf-8"))

        split_text = text.rsplit("\n", 1)[1]
        to_json = json.loads(split_text)
        return to_json
