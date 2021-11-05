# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from services import recognizers_pb2 as recognizers__pb2


class RecognizersTextStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetRecognizeResult = channel.unary_unary(
        '/RecognizersText/GetRecognizeResult',
        request_serializer=recognizers__pb2.TextRequest.SerializeToString,
        response_deserializer=recognizers__pb2.RecognizeResult.FromString,
        )


class RecognizersTextServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def GetRecognizeResult(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_RecognizersTextServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetRecognizeResult': grpc.unary_unary_rpc_method_handler(
          servicer.GetRecognizeResult,
          request_deserializer=recognizers__pb2.TextRequest.FromString,
          response_serializer=recognizers__pb2.RecognizeResult.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'RecognizersText', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
