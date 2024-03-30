import os
import requests


def scrape_linkedin_profile(linkedin_profile_url: str = None):
    """
    Scrape information from linkedin profiles, 
    Manually scrape the information from LinkedIn profile
    """
    api_endpoint = None
    header_dict = None
    response = None

    # You can use the LinkedIn API from proxycurl at: https://nubela.co/proxycurl/docs#people-api-person-lookup-endpoint
    # Since it has limited credits, already got the information and stored it in gist.github.com
    if linkedin_profile_url:
        api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
        header_dict = {
            "Authorization": f"Bearer {os.getenv('PROXYCURL_API_KEY')}"}
        response = requests.get(url=api_endpoint, params={
                                "linkedin_profile_url": linkedin_profile_url}, headers=header_dict)
    else:
        api_endpoint = 'https://gist.githubusercontent.com/<account_name>/<some_guid>/raw/<some_guid>/<account_name>.json'
        header_dict = {
            "Authorization": f"token {os.getenv('GITHUB_TOKEN')}", "Accept": "application/vnd.github.v4+raw"}
        response = requests.get(url=api_endpoint, headers=header_dict)

    if response and response.status_code == 200:
        data = response.json()

        # Remove unncessary information and keep it compact
        # This is especially needed for smartly using token limits.
        data = {
            k: v
            for k, v in data.items()
            if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
        }
        if data.get("groups"):
            for group_dict in data.get("groups"):
                group_dict.pop("profile_pic_url")

        return data
    else:
        print(f"Error in response: {response}")
