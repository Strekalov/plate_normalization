from locust import HttpUser, TaskSet, between, task


class UserBehavior(TaskSet):
    @task
    def post_request(self):
        with open("assets/1a8fd53a448467ad.jpg", "rb") as file:
            files = {"image": ("fdc3e7b9662f7229.jpg", file, "image/jpeg")}
            headers = {
                "accept": "application/json",
            }
            self.client.post(
                "/plates/get_normalized_image_v1", files=files, headers=headers
            )


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(0, 0.1)  # Настройте время ожидания между запросами


# Слушатели для начала и конца теста
from locust import events


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Тест начался")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Тест завершен")
