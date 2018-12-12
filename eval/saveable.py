from __future__ import print_function

import io
from PIL import Image
from azure.storage.blob import BlockBlobService

class Saveable(object):
    def save(self, relpath, name, image):
        raise NotImplementedError

class BlobSaveable(Saveable):
    def __init__(self, account_name, account_key, container_name):
        self.container_name = container_name
        self.blob_service = BlockBlobService(account_name=account_name, account_key=account_key)

    def save(self, relpath, name, image):
        # save image to bytes stream
        im = Image.fromarray(image)
        stream = io.BytesIO()
        im.save(stream, format='png')
        stream.seek(0)

        # upload blob
        blob_name = '{}/{}.png'.format(relpath, name)
        self.blob_service.create_blob_from_stream(self.container_name, blob_name, stream)

        # return path
        path = '{}://{}/{}/{}'.format(self.blob_service.protocol, 
                                      self.blob_service.primary_endpoint, 
                                      self.container_name, blob_name)
        return path
